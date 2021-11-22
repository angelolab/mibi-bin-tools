from typing import Any, Dict, List, Tuple, Union
import os
import json
import multiprocessing as mp

import numpy as np
import pandas as pd
import skimage.io as io

from mibi_bin_tools import io_utils, _extract_bin


def _mass2tof(masses_arr, mass_offset, mass_gain, time_res):
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


def _set_tof_ranges(fov: Dict[str, Any], higher: np.ndarray, lower: np.ndarray, time_res: float):
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
            Fovs argument is modified
    """
    key_names = ('upper_tof_range', 'lower_tof_range')
    mass_ranges = (higher, lower)
    wrapping_functions = (np.ceil, np.floor)

    for key, masses, wrap in zip(key_names, mass_ranges, wrapping_functions):
        fov[key] = \
            wrap(
                _mass2tof(masses, fov['mass_offset'], fov['mass_gain'], time_res)
            ).astype(np.uint16)


def _write_out(img_data, out_dir, fov_name, targets):
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
    """
    out_dirs = [
        os.path.join(out_dir, fov_name),
        os.path.join(out_dir, fov_name, 'intensities'),
        os.path.join(out_dir, fov_name, 'intensity_times_width')
    ]
    suffixes = [
        '',
        '_intensity',
        '_int_width'
    ]
    for i, out_dir, suffix in enumerate(zip(out_dirs, suffixes)):
        if not os.path.exitst(out_dir):
            os.makedirs(out_dir)
        for j, target in enumerate(targets):
            io.imsave(os.path.join(out_dir, f'{target}{suffix}.tiff'), img_data[i, :, :, j],
                      plugin='tifffile', check_contrast=False)


def _find_bin_files(data_dir: str, include_fovs: Union[List[str], None] = None):
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
        fov_files = {fov_file: fov_files[fov_file] for fov_file in include_fovs}
    
    return fov_files


def _fill_fov_metadata(data_dir: str, fov: Dict[str, Any],
                       panel: Union[Tuple[float, float], pd.DataFrame],
                       intensities: Union[bool, List[str]], time_res: float,
                       channels: List[str] = None):
 
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
                        time_res: float, channels: List[str]):

    rows = json_metadata['fov']['panel']['conjugates']
    fov['masses'], fov['targets'] = zip(*[
        (el['mass'], el['target'])
        for el in rows
        if channels is None or el['target'] in channels
    ])

    masses_arr = np.array(fov['masses'])
    _set_tof_ranges(fov, masses_arr + panel[1], masses_arr + panel[0], time_res)


def _parse_df_panel(fov: Dict[str, Any], panel: pd.DataFrame, time_res: float, channels: List[str]):

    rows = panel.loc[panel['Target'].isin(panel['Target'] if channels is None else channels)]
    fov['masses'] = rows['Mass']
    fov['targets'] = rows['Target']

    _set_tof_ranges(fov, rows['Stop'].values, rows['Start'].values, time_res)


def _parse_intensities(fov: Dict[str, Any], intensities: Union[bool, List[str]]):

    if type(intensities) is list:
        fov['intensities'] = [target in intensities for target in fov['targets']]
    elif intensities is True:
        fov['intensities'] = fov['targets']

    # order the 'calc_intensity' bools 
    if 'intensities' in fov.keys():
        fov['calc_intensity'] = [target in fov['intensities'] for target in fov['targets']]
    else:
        fov['calc_intensity'] = [False,] * len(fov['targets'])


def extract_bin_files(data_dir: str, out_dir: str,
                      include_fovs: Union[List[str], None] = None,
                      panel: Union[Tuple[float, float], pd.DataFrame] = (-0.3, 0.0),
                      intensities: Union[bool, List[str]] = False, time_res: float=500e-6):
    """Converts MibiScope bin files to pulse count, intensity, and intensity * width tiff images
    
    Args:
        data_dir (str | PathLike):
            Directory containing bin files as well as accompanying json metadata files
        out_dir (str | PathLike):
            Directory to save the tiffs in
        include_fovs (List | None):
            List of fovs to include.  Includes all if None.
        panel (tuple | pd.DataFrame):
            If a tuple, global integration range over all antibodies within json metadata.
            If a pd.DataFrame, specific peaks with custom integration ranges.  Column names must be
            'Mass' and 'Target' with integration ranges specified via 'Start' and 'Stop' columns.
        intensities (bool | List):
            Whether or not to extract intensity and intensity * width images.  If a List, specific
            peaks can be extracted, ignoring the rest, which will only have pulse count images
            extracted.
        time_res (float):
            Time resolution for scaling parabolic transformation
    """
    fov_files = _find_bin_files(data_dir, include_fovs)

    for fov in fov_files.values():
        _fill_fov_metadata(data_dir, fov, panel, intensities, time_res)

    bin_files = \
        [(fov, os.path.join(data_dir, fov['bin'])) for fov in fov_files.values()]

    with mp.Pool() as pool:
        for fov, bf in bin_files:
            # call extraction cython here
            img_data = _extract_bin.c_extract_bin(
                bytes(bf, 'utf-8'), fov['lower_tof_range'],
                fov['upper_tof_range'], np.array(fov['calc_intensity'], dtype=np.uint8))
            pool.apply_async(
                _write_out, 
                (img_data, out_dir, fov['bin'][:-4], fov['targets'])
            )
        pool.close()
        pool.join()

def get_width_histogram(data_dir: str, out_dir: str, fov: str, channel: str,
                        mass_range=(-0.3, 0.0), time_res: float=500e-6):
    """Generates histogram of all pulse widths found within the given mass range

    
    Args:
        data_dir (str | PathLike):
            Directory containing bin files as well as accompanying json metadata files
        out_dir (str | PathLike):
            Directory to save the tiffs in
        fov (str):
            Fov to generate histogram for
        channel (str):
            Channel to check widths for
        mass_range (tuple):
            Integration range
        time_res (float):
            Time resolution for scaling parabolic transformation
    """
    fov = _find_bin_files(data_dir, [fov])[0]
    
    _fill_fov_metadata(data_dir, fov, mass_range, False, time_res, [channel])

    local_bin_file = os.path.join(data_dir, fov['bin'])

    discovered = _extract_bin.c_extract_no_sum(bytes(local_bin_file, 'utf-8'), 
                                               fov['lower_tof_range'],
                                               fov['upper_tof_range'])

    return discovered
