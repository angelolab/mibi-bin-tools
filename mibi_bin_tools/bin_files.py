from typing import Any, Dict, List, Tuple, Union
import os
import json
import multiprocessing as mp

import numpy as np
import pandas as pd
import skimage.io as io
import collections
from sympy import symbols, solve , Eq

from mibi_bin_tools import io_utils, _extract_bin

def mass2tof(masses_arr, mass_offset, mass_gain, time_res):
    return (mass_gain * np.sqrt(masses_arr) + mass_offset) / time_res

def tof2mass(time_arr, mass_offset, mass_gain, time_res: float=500e-6):
    return ((time_arr * time_res - mass_offset) / mass_gain) ** 2

def _set_tof_ranges(fov: Dict[str, Any], higher: np.ndarray, lower: np.ndarray, time_res: float):
    key_names = ('upper_tof_range', 'lower_tof_range')
    mass_ranges = (higher, lower)
    wrapping_functions = (np.ceil, np.floor)

    for key, masses, wrap in zip(key_names, mass_ranges, wrapping_functions):
        fov[key] = \
            wrap(
                mass2tof(masses, fov['mass_offset'], fov['mass_gain'], time_res)
            ).astype(np.uint16)

def write_out(img_data, intensity_data, intens_width_data, out_dir, fov_name, targets):
    final_out = os.path.join(out_dir, fov_name)
    int_out = os.path.join(final_out, 'intensities')
    int_width_out = os.path.join(final_out, 'intensity_times_width')
    os.makedirs(final_out)
    if not os.path.exists(int_out):
        os.makedirs(int_out)
        os.makedirs(int_width_out)

    for i, target in enumerate(targets):
        io.imsave(os.path.join(final_out, f'{target}.tiff'), img_data[:, :, i].astype(np.uint16), plugin='tifffile', check_contrast=False)
        if np.max(intensity_data) != 0:
            io.imsave(os.path.join(int_out, f'{target}_intensity.tiff'), intensity_data[:, :, i], plugin='tifffile', check_contrast=False)
            io.imsave(os.path.join(int_width_out, f'{target}_int_width.tiff'), intens_width_data[:, :, i], plugin='tifffile', check_contrast=False)
            
def write_out_spectra(pd, out_dir, fov_name):
    final_out = os.path.join(out_dir, 'spectrum_files')
    if not os.path.exists(final_out):
        os.makedirs(final_out)
    pd.to_csv(os.path.join(final_out, f'{fov_name}_spectrum.csv'))

def get_spectra_df(TimeOffset):
    data = collections.Counter(TimeOffset)
    df = pd.DataFrame(list(zip(data.keys(), data.values())), columns =['TimeOffset', 'Counts'])
    df = df.sort_values(by=['TimeOffset'])

    ##add padded array where TOFs are empty
    timePadded = np.arange(0 , np.max(df['TimeOffset']) + 1 , 1)
    df_padded = pd.DataFrame(data= timePadded , columns =['TimeOffset'])
    df_padded = df_padded.merge(df, how='left', on='TimeOffset')
    df_padded.loc[df_padded.Counts.isna() , 'Counts'] = 0
    
    return df_padded 

def calibrate_spectrum(t1 , m1 , t2 , m2 , time_res: float=500e-6):
    a, b = symbols('a b')
    eq1 = Eq((((t1 * time_res - b)/a) ** 2 - m1) , 0)
    eq2 = Eq((((t2 * time_res - b)/a) ** 2 - m2) , 0)
    sol = solve((eq1 , eq2) , (a , b))
    #get the negative mass_offset (???)
    ind = [i for i in range(len(sol)) if sol[i][1] < 0 and sol[i][0] > 0 ]
    mass_offset = sol[ind[0]][1]
    mass_gain = sol[ind[0]][0]
    return mass_offset , mass_gain
        
def extract_bin_files(data_dir: str, out_dir: str,
                      include_fovs: Union[List[str], None] = None,
                      panel: Union[Tuple[float, float], pd.DataFrame] = (-0.3, 0.0),
                      intensities: Union[bool, List[str]] = False, time_res: float=500e-6,
                      timeout=100 , calibration: Union[Tuple[float, float], str] = 'auto'):
    
    # TODO: intensities

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

    for fov in fov_files.values():
        with open(os.path.join(data_dir, fov['json']), 'rb') as f:
            data = json.load(f)

        fov['mass_gain'] = data['fov']['fullTiming']['massCalibration']['massGain']
        fov['mass_offset'] = data['fov']['fullTiming']['massCalibration']['massOffset']

        if type(calibration) is tuple:
            fov['mass_gain'] = calibration[1]
            fov['mass_offset'] = calibration[0]
        elif calibration == 'auto':
            fov['mass_gain'] = data['fov']['fullTiming']['massCalibration']['massGain']
            fov['mass_offset'] = data['fov']['fullTiming']['massCalibration']['massOffset']
            
        if type(panel) is tuple:
            rows = data['fov']['panel']['conjugates']
            fov['masses'], fov['targets'] = zip(*[(el['mass'], el['target']) for el in rows])

            masses_arr = np.array(fov['masses'])
            _set_tof_ranges(fov, masses_arr + panel[1], masses_arr + panel[0], time_res)
        else:
            fov['masses'] = panel['Mass']
            fov['targets'] = panel['Target']

            _set_tof_ranges(fov, panel['Stop'].values, panel['Start'].values, time_res)

        if type(intensities) is list:
            fov['intensities'] = [target in intensities for target in fov['targets']]
        elif intensities is True:
            fov['intensities'] = fov['targets']

        # order the 'calc_intensity' bools 
        if 'intensities' in fov.keys():
            fov['calc_intensity'] = [target in fov['intensities'] for target in fov['targets']]
        else:
            fov['calc_intensity'] = [False,] * len(fov['targets'])

    # start download of bin files to new tmp dir
    bin_file_paths = [os.path.join(data_dir, fov['bin']) for fov in fov_files.values()]
    bin_file_sizes = [os.path.getsize(bfp) for bfp in bin_file_paths]
    bin_files = \
        [(fov, os.path.join(data_dir, fov['bin'])) for fov in fov_files.values()]

    with mp.Pool() as pool:
        for i, (fov, bf) in enumerate(bin_files):
            # call extraction cython here
            img_data = _extract_bin.c_extract_bin(
                bytes(bf, 'utf-8'), bin_file_sizes[i], fov['lower_tof_range'],
                fov['upper_tof_range'], np.array(fov['calc_intensity'], dtype=np.uint8),
                timeout=timeout
            )
            pool.apply_async(
                write_out, 
                (img_data[0, :, :, :], img_data[1, :, :, :], img_data[2, :, :, :], out_dir,
                 fov['bin'][:-4], fov['targets']
                )
            )
        pool.close()
        pool.join()

def extract_no_sum(data_dir, out_dir, fov, channel, mass_range=(-0.3, 0.0), time_res: float=500e-6):
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

    fov = fov_files[fov]
    with open(os.path.join(data_dir, fov['json']), 'rb') as f:
        data = json.load(f)

    fov['mass_gain'] = data['fov']['fullTiming']['massCalibration']['massGain']
    fov['mass_offset'] = data['fov']['fullTiming']['massCalibration']['massOffset']

    rows = data['fov']['panel']['conjugates']
    fov['masses'], fov['targets'] = zip(*[(el['mass'], el['target']) for el in rows])
    t_index = fov['targets'].index(channel)
    fov['masses'] = fov['masses'][t_index]
    fov['targets'] = fov['targets'][t_index]

    masses_arr = np.array(fov['masses'])
    _set_tof_ranges(fov, masses_arr + mass_range[1], masses_arr + mass_range[0], time_res)

    # start download of bin files to new tmp dir
    bin_file_paths = os.path.join(data_dir, fov['bin'])
    bin_file_sizes = os.path.getsize(bin_file_paths)

    local_bin_file = os.path.join(data_dir, fov['bin'])

    with mp.Pool() as pool:
        # call extraction cython here
        discovered = _extract_bin.c_extract_no_sum(bytes(local_bin_file, 'utf-8'), bin_file_sizes,
                                                   fov['lower_tof_range'],
                                                   fov['upper_tof_range'], 100
        )
        pool.close()
        pool.join()

    return discovered

def extract_spectra(data_dir: str, out_dir: str,
                      include_fovs: Union[List[str], None] = None,
                      panel: Union[Tuple[float, float], pd.DataFrame] = (-0.3, 0.0),
                      intensities: Union[bool, List[str]] = False, time_res: float=500e-6,
                      timeout=100 , calibration: Union[Tuple[float, float], str] = 'auto'):
    
    # TODO: intensities

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

    for fov in fov_files.values():
        with open(os.path.join(data_dir, fov['json']), 'rb') as f:
            data = json.load(f)

        fov['mass_gain'] = data['fov']['fullTiming']['massCalibration']['massGain']
        fov['mass_offset'] = data['fov']['fullTiming']['massCalibration']['massOffset']

        if type(calibration) is tuple:
            fov['mass_gain'] = calibration[1]
            fov['mass_offset'] = calibration[0]
        elif calibration == 'auto':
            fov['mass_gain'] = data['fov']['fullTiming']['massCalibration']['massGain']
            fov['mass_offset'] = data['fov']['fullTiming']['massCalibration']['massOffset']
            
        if type(panel) is tuple:
            rows = data['fov']['panel']['conjugates']
            fov['masses'], fov['targets'] = zip(*[(el['mass'], el['target']) for el in rows])

            masses_arr = np.array(fov['masses'])
            _set_tof_ranges(fov, masses_arr + panel[1], masses_arr + panel[0], time_res)
        else:
            fov['masses'] = panel['Mass']
            fov['targets'] = panel['Target']

            _set_tof_ranges(fov, panel['Stop'].values, panel['Start'].values, time_res)

        if type(intensities) is list:
            fov['intensities'] = [target in intensities for target in fov['targets']]
        elif intensities is True:
            fov['intensities'] = fov['targets']

        # order the 'calc_intensity' bools 
        if 'intensities' in fov.keys():
            fov['calc_intensity'] = [target in fov['intensities'] for target in fov['targets']]
        else:
            fov['calc_intensity'] = [False,] * len(fov['targets'])

    # start download of bin files to new tmp dir
    bin_file_paths = [os.path.join(data_dir, fov['bin']) for fov in fov_files.values()]
    bin_file_sizes = [os.path.getsize(bfp) for bfp in bin_file_paths]
    bin_files = \
        [(fov, os.path.join(data_dir, fov['bin'])) for fov in fov_files.values()]

    with mp.Pool() as pool:
        for i, (fov, bf) in enumerate(bin_files):
            # call extraction cython here
            spectra_data = _extract_bin.c_extract_spectra(
                bytes(bf, 'utf-8'), bin_file_sizes[i], fov['lower_tof_range'],
                fov['upper_tof_range'], np.array(fov['calc_intensity'], dtype=np.uint8),
                timeout=timeout
            )
            #build spectra from pulses list    
            df_spectra = get_spectra_df(spectra_data)
            df_spectra['massList'] = pd.Series(tof2mass(df_spectra['TimeOffset'].to_numpy(), fov['mass_offset'], fov['mass_gain'], time_res))
            #save to spectra.csv
            pool.apply_async(
                write_out_spectra, 
                (df_spectra, out_dir, fov['bin'][:-4]
                )
            )
        pool.close()
        pool.join()
    
    return df_spectra