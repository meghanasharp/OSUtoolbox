#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for the radar processing.

@author: Kirill Ivanov
"""

from impdar.lib import load, plot
from impdar.lib.gpslib import interp
import matplotlib.pyplot as plt
import re
import os
import numpy as np
import pandas as pd

def file_process(file, MHz, migration=False, save=False, plot_figure = False):
    """
    Fuction for bulk process lines within file
    Code by Kirill Ivanov

    Parameters
    ----------
    data: impdar.lib.RadarData.RadarData
        full hdf5 file, radar data with all lines

    MHz: int
        input for signal output used for different band passes.

    migration: False/True
        True if you want to run phsh migration

    save: False/True
        input True if want to save processed data in .mat format

    plot: False/True
        input True if want to plot radargram and save the figure

    Return
    ----------
    proc_data: dict
        A dictionarry of processed lines, can be indexted by line name
    """

    for line_number in range(len(file)):
        data = file[line_number]
        file_name = re.split('/',vars(data)['fn'])[-1]
        full_name = file_name.split('.')[0]
        print('Analyzing ' + full_name)
        data.long = -1 * data.long
        data.crop(0.,dimension='pretrig', rezero = True)
        data.nmo(ant_sep = 5)
        data.winavg_hfilt(avg_win = 301,taper='full')
        if MHz == 5:
            data.vertical_band_pass(1,15)
        elif MHz == 20:
            data.vertical_band_pass(3,30)
        else:
            raise ValueError('Incorrect MHz input. Add bandpass for the desired MHz input.')
        interp([data],3)
        data.denoise(3,5)
        data.rangegain(0.007)
        #data.elev_correct()
        #data.reverse() -- reverse for along the flow lines? how to get that info
        #data.hcrop_ki(3686,4120)
        if migration:
            data.migrate(mtype='phsh')
        data.elev_correct()
        if save:
            name = full_name + '.mat'
            data.save(name)
        if plot_figure:
            plot.plot_radargram(data,ydat='depth',xdat='dist')
            #name = full_name + '.png'
            #plt.savefig(name)
        if line_number == 0:
            proc_data = {full_name: data}
        else:
            new_dict = {full_name: data}
            proc_data.update(new_dict)
    return proc_data

def z_correction(z_corrected_csv):
    """
    Fuction for bulk assign corrected elevation for lines within file and save it as .mat file
    Code by Kirill Ivanov

    Parameters
    ----------

    {Required}
    Funciton will run in the current working directory for every hdf5 file.
    Make sure that only IceRadar hdf5 files are in the directory.

    z_corrected_csv: csv
        A csv file/table with corrected z-elevation as z_interp,
                                        file name as "File Name",
                                        Line Name as "Line Name",
                                        and lat as "Y - lat"

    Return
    ----------

    None. Saves each line with corrected elevation as new .mat file.
    """
    table = pd.read_csv(z_corrected_csv,
                        index_col = 'datetime')

    for file in os.listdir(os.getcwd()):
        if file.endswith('.hdf5'):
            print('Processing ' + file + '.')
            f_in = load.load('bsi',file)
            for line in  f_in:
                name = re.split('/',vars(line)['fn'])[-1]
                file_name = name.split('line')[0] + '.hdf5'
                line_name = 'line' + re.split('line',name)[1].split('.')[0]
                line.long = -1*line.long
                index = (table['file_name'] == file_name)&(table['line_name'] == line_name)
                line.elev = table[index].z_interp.to_numpy()
                final_name = file_name.split('.')[0] + '_' + line_name + '.mat'
                line.save(final_name)
        else:
            print(file + ' is not supported. Only .hdf5 are accepted.')
    return


def power_csv(bulk = True, file_single = None):
    """
    Funciton that will compile power for all lines into a single csv file.
        Also includes lat and lon for plotting.
    Code by Kirill Ivanov

    Parameters
    ----------
    bulk: str
        True : process all .mat file in the current working directory.
                CHECK THAT ALL .mat FILE HAVE BED PICK
        False : process a single file.

    Optional:
    file_single : .mat file
        If {bulk= False} chosen, provide .mat file that has a single pick of the bed.

    Returns
    -------
    None.

    """

    lat = []
    lon = []
    power = []
    def power_process(file,lat,lon,power):
        f_in = load.load('mat',file)[0]
        lat = np.append(lat,vars(f_in)['lat'])
        lon = np.append(lon,vars(f_in)['long'])
        pp = f_in.picks.power
        for i in range(len(pp)):
            if np.all(np.isnan(pp[i])) == False:
                power = np.append(power,pp[i])
        return lat,lon,power

    if bulk:
        for file in os.listdir(os.getcwd()):
            if file.endswith('.mat'):
                lat,lon,power = power_process(file, lat, lon, power)
            else:
                raise ValueError('Inappropriate parameter. Choose .mat')
    else:
        if file_single:
            if file_single.endswith('.mat'):
                lat,lon,power = power_process(file_single, lat, lon, power)
            else:
                raise ValueError('Inappropriate parameter. Choose .mat')
        else:
            raise ValueError('Needs a file name for single file processing.')
    final = {'Lat': lat,
             'Long': lon,
             'Power': power
        }
    df = pd.DataFrame(data=final,index = None)
    df.to_csv('Power.csv')
    return