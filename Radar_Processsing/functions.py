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

#%%
def file_process(file, MHz, migration=False, save=False):
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
        input True is want to save processed data in .mat format
        
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
        data.crop(0.,dimension='pretrig',rezero = 'True')
        data.nmo(ant_sep = 2)
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
        #data.reverse() -- reverse for along the flow lines? how to get that info
        #data.hcrop_ki(3686,4120)
        if migration:
            data.migrate(mtype='phsh')
        #plot.plot_radargram(data,ydat='depth',xdat='dist')
        #plt.show()
        if save:
            name = full_name + '.mat'
            data.save(name)
        plot.plot_radargram(data,ydat='depth',xdat='dist')
        name = full_name + '.png'
        plt.savefig(name)
        if line_number == 0:
            proc_data = {full_name, data}
        else:
            new_dict = {full_name, data}
            proc_data.update(new_dict) 
    return proc_data

def power_csv(bulk = 'all', file = None):
    """
    Funciton that will compile power for all lines into a single csv file. 
        Also includes lat and lon for plotting. 
    Code by Kirill Ivanov

    Parameters
    ----------
    bulk: str
        'all': process all .mat file in the current working directory. 
                CHECK THAT ALL .mat FILE HAVE BED PICK
        'single': process a single file. 
        
    Optional:
    file : .mat file 
        If {bulk='single'} chosen, provide .mat file that has a single pick of the bed.

    Returns
    -------
    None.

    """
    #get all .mat files if bulk
    #make option for non bulk just 1 file 
    #load.load mat file 
    #change long for this time 
    #check picks dat.picks.power[i] which one has data np.any ~isnan
    # get that array into pandas dataframe 
    #repeat for each file
    #save csv file with lat long power
    lat = []
    lon = []
    power = []
    
    for file in os.listdir(os.getcwd()):
        if file.endswith('.mat'):
            f_in = load.load('mat',file)[0]
            #with load.load('mat',file)[0] as f_in:
            lat = np.append(lat,vars(f_in)['lat'])
            lon = np.append(lon,-1 * vars(f_in)['long'])
            pp = f_in.picks.power
            for i in range(len(pp)):
                if np.all(np.isnan(pp[i])) == False:
                    power = np.append(power,pp[i])
        else:
            raise ValueError('Inappropriate parameter. Choose ')
    final = {'Lat': lat,
             'Long': lon,
             'Power': power
        }
    df = pd.DataFrame(data=final,index = None)
    df.to_csv('Power.csv')
    return