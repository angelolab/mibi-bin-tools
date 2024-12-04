from typing import Any, Dict, List, Tuple, Union
import os
import json

import numpy as np
import pandas as pd
import skimage.io as io
import xarray as xr

from mibi_bin_tools import type_utils, _extract_bin
from alpineer import io_utils, image_utils


def _mass2tof(masses_arr: np.ndarray, mass_offset: float, mass_gain: float,
              time_res: float) -> np.ndarray:
    """Convert array of m/z values to equivalent time of flight values

    Args:
        masses_arr (array_like):
            Array of m/z values
        mass_offset (float):
            Mass offset for parabolic transformation
        mass_gain (float):
            Mass gain for parabolic transformation
        time_res (float):
            Time resolution for scaling parabolic transformation

    Returns:
        array_like:
            Array of time of flight values; indicies paried to `masses_arr`
    """
    return (mass_gain * np.sqrt(masses_arr) + mass_offset) / time_res


def _tof2mass(tof_arr: np.ndarray, mass_offset: float, mass_gain: float,
              time_res: float) -> np.ndarray:
    """Convert array of time of flight values to equivalent m/z

    Args:
        tof_arr (array_like):
            Array of time of flight values
        mass_offset (float):
            Mass offset for parabolic transformation
        mass_gain (float):
            Mass gain for parabolic transformation
        time_res (float):
            Time resolution for scaling parabolic transformation

    Returns:
        array_like:
            Array of m/z values; indicies paried to `tof_range`
    """
    return (((time_res * tof_arr) - mass_offset) / mass_gain) ** 2


def _set_tof_ranges(fov: Dict[str, Any], higher: np.ndarray, lower: np.ndarray,
                    time_res: float) -> None:
    """Converts and stores provided mass ranges as time of flight ranges within fov metadata

    Args:
        fov (Dict[str, Any]):
            Metadata for the fov.
        higher (array_like):
            Array of m/z values; upper bounds for integration
        lower (array_like):
            Array of m/z values; lower bounds for integration
        time_res (float):
            Time resolution for scaling parabolic transformation

    Returns:
        None:
            Fovs argument is modified in place
    """
    key_names = ('upper_tof_range', 'lower_tof_range')
    mass_ranges = (higher, lower)

    for key, masses in zip(key_names, mass_ranges):
        # truncate the conversion to ensure consistency
        fov[key] = _mass2tof(
            masses, fov['mass_offset'], fov['mass_gain'], time_res
        ).astype(np.uint16)


def _write_out(img_data: np.ndarray, out_dir: str, fov_name: str, targets: List[str],
               intensities: Union[bool, List[str]] = False) -> None:
    """Parses extracted data and writes out tifs

    Args:
        img_data (np.ndarray):
            Array containing the pulse counts, intensity, and intensity * width images
        out_dir (str | PathLike):
            Directory to save tifs
        fov_name (str):
            Name of the field of view
        targets (array_like):
            List of target names (i.e channels)
        intensities (bool | List):
            Whether or not to write out intensity images.  If a List, specific
            peaks can be written out, ignoring the rest, which will only have pulse count images.
    """
    out_dirs = [
        os.path.join(out_dir, fov_name),
        os.path.join(out_dir, fov_name, 'intensities'),
    ]
    suffixes = [
        '',
        '_intensity',
    ]
    save_dtypes = [
        np.uint32,
        np.uint32,
    ]

    for i, (out_dir_i, suffix, save_dtype) in enumerate(zip(out_dirs, suffixes, save_dtypes)):
        # break loop when index is larger than type dimension of img_data
        if i+1 > img_data.shape[0]:
            break
        if not os.path.exists(out_dir_i):
            os.makedirs(out_dir_i)
        for j, target in enumerate(targets):
            # save all first images regardless of replacing
            # if not replace (i=1), only save intensity images for specified targets
            if i == 0 or (target in list(intensities)):
                fname = os.path.join(out_dir_i, f"{target}{suffix}.tiff")
                image_utils.save_image(fname=fname, data=img_data[i, :, :, j].astype(save_dtype))


def _find_bin_files(data_dir: str,
                    include_fovs: Union[List[str], None] = None) -> Dict[str, Dict[str, str]]:
    """Locates paired bin/json files within the provided directory.

    Args:
        data_dir (str | PathLike):
            Directory containing bin/json files
        include_fovs (List | None):
            List of fovs to include. Includes all if None.

    Returns:
        Dict[str, Dict[str, str]]:
            Dictionary containing the names of the valid bin files
    """
    bin_files = io_utils.list_files(data_dir, substrs=['.bin'])
    json_files = io_utils.list_files(data_dir, substrs=['.json'])

    fov_names = io_utils.extract_delimited_names(bin_files, delimiter='.')

    fov_files = {
        fov_name: {
            'bin': fov_name + '.bin',
            'json': fov_name + '.json',
        }
        for fov_name in fov_names
        if fov_name + '.json' in json_files
    }

    if include_fovs is not None:
        fov_files = {
            fov_file: fov_files[fov_file]
            for fov_file in include_fovs
            if fov_file in fov_files
        }

    if not len(fov_files):
        raise FileNotFoundError(f'No viable bin files were found in {data_dir}...')

    return fov_files


def _fill_fov_metadata(data_dir: str, fov: Dict[str, Any],
                       panel: Union[Tuple[float, float], pd.DataFrame],
                       intensities: Union[bool, List[str]], time_res: float,
                       channels: List[str] = None) -> None:
    """ Parses user input and mibiscope json to build extraction parameters

    Fills fov metadata with mass calibration parameters, builds panel, and sets intensity
    extraction flags.

    Args:
        data_dir (str):
            Directory containing bin files as well as accompanying json metadata files
        fov (Dict[str, Any]):
            Metadata for the fov.
        panel (tuple | pd.DataFrame):
            If a tuple, global integration range over all antibodies within json metadata.
            If a pd.DataFrame, specific peaks with custom integration ranges.  Column names must be
            'Mass' and 'Target' with integration ranges specified via 'Start' and 'Stop' columns.
        intensities (bool | List[str]):
            Whether or not to extract intensity and intensity * width images.  If a List, specific
            peaks can be extracted, ignoring the rest, which will only have pulse count images
            extracted.
        time_res (float):
            Time resolution for scaling parabolic transformation
        channels (List[str] | None):
            Filters panel for given channels.  All channels in panel extracted if None
    Returns:
        None:
            `fov` argument is modified in place
    """

    with open(os.path.join(data_dir, fov['json']), 'rb') as f:
        data = json.load(f)

    fov['mass_gain'] = data['fov']['fullTiming']['massCalibration']['massGain']
    fov['mass_offset'] = data['fov']['fullTiming']['massCalibration']['massOffset']

    if type(panel) is tuple:
        _parse_global_panel(data, fov, panel, time_res, channels)
    else:
        _parse_df_panel(fov, panel, time_res, channels)

    _parse_intensities(fov, intensities)


def _parse_global_panel(json_metadata: dict, fov: Dict[str, Any], panel: Tuple[float, float],
                        time_res: float, channels: List[str]) -> None:
    """Extracts panel contained in mibiscope json metadata

    Args:
        json_metadata (dict):
            metadata read via mibiscope json
        fov (Dict[str, Any]):
            Metadata for the fov.
        panel (tuple):
            Global integration range over all antibodies within json metadata.
            Column names must 'Mass' and 'Target' with integration ranges specified via 'Start' and
            'Stop' columns.
        time_res (float):
            Time resolution for scaling parabolic transformation
        channels (List[str] | None):
            Filters panel for given channels.  All channels in panel extracted if None
    Returns:
        None:
            `fov` argument is modified in place
    """
    if json_metadata['fov'].get('panel', None) is None:
        raise KeyError(
            f"'panel' field not found in {fov['json']}. "
            + "If this is a moly point, you must manually supply a panel..."
        )
    rows = json_metadata['fov']['panel']['conjugates']
    fov['masses'], fov['targets'] = zip(*[
        (el['mass'], el['target'])
        for el in rows
        if channels is None or el['target'] in channels
    ])

    masses_arr = np.array(fov['masses'])
    _set_tof_ranges(fov, masses_arr + panel[1], masses_arr + panel[0], time_res)


def _parse_df_panel(fov: Dict[str, Any], panel: pd.DataFrame, time_res: float,
                    channels: List[str]) -> None:
    """Converts masses from panel into times for fov extraction-metadata structure

    Args:
        fov (Dict[str, Any]):
            Metadata for the fov.
        panel (pd.DataFrame):
            Specific peaks with custom integration ranges.  Column names must be 'Mass' and
            'Target' with integration ranges specified via 'Start' and 'Stop' columns.
        time_res (float):
            Time resolution for scaling parabolic transformation
        channels (List[str] | None):
            Filters panel for given channels.  All channels in panel extracted if None
    Returns:
        None:
            `fov` argument is modified in place
    """
    rows = panel.loc[panel['Target'].isin(panel['Target'] if channels is None else channels)]
    fov['masses'] = rows['Mass']
    fov['targets'] = rows['Target']

    _set_tof_ranges(fov, rows['Stop'].values, rows['Start'].values, time_res)


def _parse_intensities(fov: Dict[str, Any], intensities: Union[bool, List[str]]) -> None:
    """Sets intensity extraction flags within the extraction-metadata

    Args:
        fov (Dict[str, Any]):
            Metadata for the fov
        intensities (bool | List):
            Whether or not to extract intensity and intensity * width images.  If a List, specific
            peaks can be extracted, ignoring the rest, which will only have pulse count images
            extracted.
    Returns:
        None:
            `fov` argument is modified in place
    """

    filtered_intensities = None
    if type(intensities) is list:
        filtered_intensities = [target for target in fov['targets'] if target in intensities]
    elif intensities is True:
        filtered_intensities = fov['targets']

    # order the 'calc_intensity' bools
    if filtered_intensities is not None:
        fov['calc_intensity'] = [target in list(filtered_intensities) for target in fov['targets']]
    else:
        fov['calc_intensity'] = [False, ] * len(fov['targets'])


def condense_img_data(img_data, targets, intensities, replace):
    """Changes image data from separate pulse and intensity data into one column if replace=True.
    Args:
        img_data (np.array):
            Contains the image data with all pulse and intensity information.
        targets (list):
            List of targets.
        intensities (bool | List):
            Whether or not to extract intensity images.  If a List, specific
            peaks can be extracted, ignoring the rest, which will only have pulse count images
            extracted.
        replace (bool):
            Whether to replace pulse images with intensity images.

    Return:
        altered img_data according to args

    """
    # extracting intensity and replacing
    if type_utils.any_true(intensities) and replace:
        for j, target in enumerate(targets):
            # replace only specified targets
            if target in intensities:
                img_data[0, :, :, j] = img_data[1, :, :, j]
        img_data = img_data[[0], :, :, :]

    # not extracting intensity
    elif not type_utils.any_true(intensities):
        img_data = img_data[[0], :, :, :]

    # extracting intensity but not replacing
    else:
        img_data = img_data[[0, 1], :, :, :]

    return img_data


def extract_bin_files(data_dir: str, out_dir: Union[str, None],
                      include_fovs: Union[List[str], None] = None,
                      panel: Union[Tuple[float, float], pd.DataFrame] = (-0.3, 0.0),
                      intensities: Union[bool, List[str]] = False, replace=True,
                      time_res: float = 500e-6):
    """Converts MibiScope bin files to pulse count, intensity, and intensity * width tiff images

    Args:
        data_dir (str | PathLike):
            Directory containing bin files as well as accompanying json metadata files
        out_dir (str | PathLike | None):
            Directory to save the tiffs in.  If None, image data is returned as an ndarray.
        include_fovs (List | None):
            List of fovs to include.  Includes all if None.
        panel (tuple | pd.DataFrame):
            If a tuple, global integration range over all antibodies within json metadata.
            If a pd.DataFrame, specific peaks with custom integration ranges.  Column names must be
            'Mass' and 'Target' with integration ranges specified via 'Start' and 'Stop' columns.
        intensities (bool | List):
            Whether or not to extract intensity images.  If a List, specific
            peaks can be extracted, ignoring the rest, which will only have pulse count images
            extracted.
        replace (bool):
            Whether to replace pulse images with intensity images.
        time_res (float):
            Time resolution for scaling parabolic transformation
    Returns:
        None | np.ndarray:
            image data if no out_dir is provided, otherwise no return
    """

    fov_files = _find_bin_files(data_dir, include_fovs)

    for fov in fov_files.values():
        _fill_fov_metadata(data_dir, fov, panel, intensities, time_res)

    bin_files = \
        [(fov, os.path.join(data_dir, fov['bin'])) for fov in fov_files.values()]

    image_data = []

    for i, (fov, bf) in enumerate(bin_files):
        img_data = _extract_bin.c_extract_bin(
            bytes(bf, 'utf-8'), fov['lower_tof_range'],
            fov['upper_tof_range'], np.array(fov['calc_intensity'], dtype=np.uint8)
        )

        # convert intensities=True to list of all targets
        if type_utils.any_true(intensities):
            if type(intensities) is not list:
                intensities = list(fov['targets'])

        img_data = condense_img_data(img_data, list(fov['targets']), intensities, replace)

        if out_dir is not None:
            _write_out(
                img_data,
                out_dir,
                fov['bin'][:-4],
                fov['targets'],
                intensities
            )
        else:
            if replace or not type_utils.any_true(intensities):
                type_list = ['pulse']
            else:
                type_list = ['pulse', 'intensities']
            image_data.append(
                xr.DataArray(
                    data=img_data[np.newaxis, :],
                    coords=[
                        [fov['bin'].split('.')[0]],
                        type_list,
                        np.arange(img_data.shape[1]),
                        np.arange(img_data.shape[2]),
                        list(fov['targets']),
                    ],
                    dims=['fov', 'type', 'x', 'y', 'channel'],
                )
            )

    if out_dir is None:
        image_data = xr.concat(image_data, dim='fov')

        return image_data


def get_histograms_per_tof(data_dir: str, fov: str, channels: List[str] = None,
                           panel: Union[Tuple[float, float], pd.DataFrame] = (-0.3, 0.0),
                           time_res: float = 500e-6):
    """Generates histograms of pulse widths, pulse counts, and pulse intensities found within the
    given mass range

    Args:
        data_dir (str | PathLike):
            Directory containing bin files as well as accompanying json metadata files
        fov (str):
            Fov to generate histogram for
        channels (str):
            Channels to check widths for, default checks all channels
        panel (tuple | pd.DataFrame):
            If a tuple, global integration range over all antibodies within json metadata.
            If a pd.DataFrame, specific peaks with custom integration ranges.  Column names must be
            'Mass' and 'Target' with integration ranges specified via 'Start' and 'Stop' columns.
        time_res (float):
            Time resolution for scaling parabolic transformation

    Returns:
        tuple (dict):
            Tuple of dicts containing widths, intensities, and pulse info per channel
    """
    fov = _find_bin_files(data_dir, [fov])[fov]

    _fill_fov_metadata(data_dir, fov, panel, False, time_res, channels)

    local_bin_file = os.path.join(data_dir, fov['bin'])

    widths, intensities, pulses = _extract_bin.c_extract_histograms(
        bytes(local_bin_file, 'utf-8'),
        fov['lower_tof_range'],
        fov['upper_tof_range']
    )

    chan_list = fov["targets"].values
    widths = {
        chan: widths_col for chan, widths_col in zip(chan_list, widths.T)
    }
    intensities = {
        chan: intensities_col for chan, intensities_col in zip(chan_list, intensities.T)
    }
    pulses = {
        chan: pulses_col for chan, pulses_col in zip(chan_list, pulses.T)
    }

    return widths, intensities, pulses


def get_median_pulse_height(data_dir: str, fov: str, channels: List[str] = None,
                            panel: Union[Tuple[float, float], pd.DataFrame] = (-0.3, 0.0),
                            time_res: float = 500e-6):
    """Retrieves median pulse intensity and mean pulse count for a given channel

    Args:
        data_dir (str | PathLike):
            Directory containing bin files as well as accompanying json metadata files
        fov (str):
            Fov to generate histogram for
        channel (str):
            Channel to check widths for
        mass_range (tuple | pd.DataFrame):
            Integration range
        time_res (float):
            Time resolution for scaling parabolic transformation

    Returns:
        dict:
            dictionary of median height values per channel
    """

    fov = _find_bin_files(data_dir, [fov])[fov]
    _fill_fov_metadata(data_dir, fov, panel, False, time_res, channels)

    local_bin_file = os.path.join(data_dir, fov['bin'])

    _, intensities, _ = \
        _extract_bin.c_extract_histograms(
            bytes(local_bin_file, 'utf-8'),
            fov['lower_tof_range'],
            fov['upper_tof_range']
        )

    int_bin = np.cumsum(intensities, axis=0) / intensities.sum(axis=0)
    median_height = (np.abs(int_bin - 0.5)).argmin(axis=0)

    chan_list = fov["targets"].values
    median_height = {
        chan: mh for chan, mh in zip(chan_list, median_height)
    }

    return median_height


def get_total_counts(data_dir: str, include_fovs: Union[List[str], None] = None):
    """Retrieves total counts for each field of view

    Args:
        data_dir (str | PathLike):
            Directory containing bin files as well as accompanying json metadata files
        include_fovs (List | None):
            List of fovs to include.  Includes all if None.

    Returns:
        dict:
            dictionary of total counts, with fov names as keys
    """

    fov_files = _find_bin_files(data_dir, include_fovs)

    bin_files = \
        [(name, os.path.join(data_dir, fov['bin'])) for name, fov in fov_files.items()]

    outs = {name: _extract_bin.c_total_counts(bytes(bf, 'utf-8')) for name, bf in bin_files}

    return outs


def get_total_spectra(data_dir: str, include_fovs: Union[List[str], None] = None,
                      panel_df: pd.DataFrame = None, range_pad: float = 0.5):
    """Retrieves total spectra for each field of view

    Args:
        data_dir (str | PathLike):
            Directory containing bin files as well as accompanying json metadata files
        include_fovs (List | None):
            List of fovs to include. Includes all if None.
        panel_df (pd.DataFrame | None):
            If not None, get default callibration information
        range_pad (float):
            Mass padding below the lowest and highest masses to consider when binning.
            The time-of-flight array go from TOF of (lowest mass - 0.5) to (highest_mass + 0.5).

    Returns:
        tuple (dict, dict, list):
            dict of total spectra and the corresponding low and high ranges, with fov names as keys
    """
    if range_pad < 0:
        raise ValueError("range_pad must be >= 0")

    fov_metadata = _find_bin_files(data_dir, include_fovs)
    if panel_df is not None:
        for fov_info in fov_metadata.values():
            _fill_fov_metadata(data_dir, fov_info, panel_df, False, 500e-6)

    bin_metadata = list(fov_metadata.items())

    # TODO: this assumes the panel_df is sorted
    lowest_mass = panel_df.loc[0, "Stop"] - range_pad
    highest_mass = panel_df.loc[panel_df.shape[0] - 1, "Stop"] + range_pad

    # store the spectra, as well as the time intervals for each FOV
    spectra = {}
    tof_interval = {}
    for fov_name, fov_info in bin_metadata:
        # compute the low and high boundaries, this will differ per FOV
        mass_offset = fov_info["mass_offset"]
        mass_gain = fov_info["mass_gain"]
        tof_boundaries = _mass2tof(
            np.array([lowest_mass, highest_mass]), mass_offset, mass_gain, 500e-6
        ).astype(np.uint16)

        # set the boundaries
        tof_interval[fov_name] = tof_boundaries

        # extract the spectra on an individual basis per channel
        spectra[fov_name] = _extract_bin.c_total_spectra(
            bytes(os.path.join(data_dir, fov_info["bin"]), "utf-8"),
            tof_boundaries[0],
            tof_boundaries[1]
        )

        # generate equivalent m/z values
        tof_arr = np.arange(tof_boundaries[0], tof_boundaries[1] + 1)
        mass_arr = _tof2mass(tof_arr, mass_offset, mass_gain, 500e-6)
        fov_info["mass_spectra_points"] = mass_arr

    return spectra, tof_interval, fov_metadata
