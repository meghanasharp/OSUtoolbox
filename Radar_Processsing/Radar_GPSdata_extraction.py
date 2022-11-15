#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
"""
Load BSI IceRadar h5 files and extract x,y,z,t.
@author: Kirill Ivanov 
"""

import re
import os
import numpy as np
import pandas as pd
import h5py
#%%
def _dm2dec(dms):
    """
    Convert the degree - decimal minute GGA to a decimal.
    Code by Kirill Ivanov 
    
    Parameters
    ----------
    dms: float
        degree-decimal minute coordinate 
        
    Return
    ----------
    dms: float
        decimal degree coordinate output 
    
    """
    if min(dms) < 0:
        mask_neg = dms < 0
        dms = abs(dms)
        dms = ((dms - dms % 100) / 100 + (dms % 100) / 60)
        dms[mask_neg] = -dms[mask_neg]
        return dms
    else:
        return ((dms - dms % 100) / 100 + (dms % 100) / 60)

#%%
def _xmlGetVal(xml, name):
    """
    Look up a value in an XML fragment. 
    Code from ImpDAR package. 
    
    Parameters
    ----------
    xml: str
        full xml file in string format (default attribute reading with h5py)
    name: str
        Variable name
        
    Return
    ----------
    ___: str
        Value of the corresponding varaible name  
     
    """
    m = re.search(r'<Name>{0}</Name>[\r]?\n<Val>'.format(
        name.replace(' ', r'\s')), xml, flags=re.IGNORECASE)
    if m is not None:
        tail = xml[m.span()[1]:]
        end = tail.find('</Val')
        return tail[:end]
    else:
        return None
#%%
def _dt_from_comment(dset):
    """
    Return the day, hour, minutes, seconds
    Code by Kirill Ivanov.
    
    Parameters
    ----------
    dset: h5py._hl.group.Group
         a single line dataset 
        
    Return
    ----------
    dmy_h_m: int
        index:
        0 - day
        1 - month
        2 - year
        3 - hour
        4 - minutes
    sec: float
        seconds 
        
    """
    t = dset.attrs['PCSavetimestamp']
    t = t[:t.find(' ')]
    split = re.split(':|_|/',t)
    dmy_h_m = list(map(int,split[:-1]))
    sec = float(split[-1])
    return [dmy_h_m[0], dmy_h_m[1],dmy_h_m[2],dmy_h_m[3],dmy_h_m[4], sec]
#%%
def _radar_xyzt_extraction(Excel=False):
    """
    Extracts x,y,z,t from hdf5 IceRadar file structure. 
    Code by Kirill Ivanov. 
    
    Parameters
    ---------- 
    
    {Required}
    None - funciton will run in the current working directory for every hdf5 file.
    Make sure that only IceRadar hdf5 files are in the directory. 
    
    {Optional}
    Excel = False/True 
        True - function will write excel file to the current directory with 
        final dataframe
    
    Return
    ----------
    df: pandas.core.frame.DataFrame 
        pandas DataFrame with following columns
        'File Name': str, 
        'Line Name': str,
        'Day': int,
        'Month': int,
        'Year': int,
        'X - lon': int, Decimal degrees coordinate
        'Y - lat': int, Decimal degrees coordinate
        'Z - Elevation': int, meter
        'Hour': int,
        'Minute': int,
        'Seconds': float. 

    """
    
    
    #Assign variables
    gps_cluster_str = 'GPS Cluster- MetaData_xml'
    gps_fix_str = 'GPS Fix valid'
    gps_message_str = 'GPS Message ok'
    alt_asl = 'Alt_asl_m'
    ch = '0'
    #Create arrays
    x = []
    y = []
    z = []
    day = []
    month = []
    year = []
    hour = []
    minute = []
    seconds = []
    F_name = []
    L_name =[]
    
    
    #check everyfile in the current working directory 
    for file in os.listdir(os.getcwd()):
        if file.endswith('.hdf5'):
            print('Processing ' + file + '.')
            #read hdf5 files
            with h5py.File(str(file),'r') as f_in:
                dset_name = [key for key in f_in.keys()]
                #read each line separetely
                for dset_name in dset_name:
                    print('Input line: ' + dset_name + ' in ' + file + '.')
                    #Create empty temporary arrays 
                    dset = f_in[dset_name]
                    tnum = len(list(dset.keys()))
                    lat = np.zeros((tnum,))
                    lon = np.zeros((tnum,))
                    elev = np.zeros((tnum,))
                    d = np.zeros((tnum,))
                    m = np.zeros((tnum,))
                    yr = np.zeros((tnum,))
                    h = np.zeros((tnum,))
                    mi = np.zeros((tnum,))
                    s = np.zeros((tnum,))
                    #get the data from every echogram in a line
                    for loc_num in range(tnum):
                        echogram = dset['location_{:d}'.format(loc_num)]['datacapture_'+ch]['echogram_'+ch]
                        # read GPS data cluster for an echogram
                        if type(dset['location_{:d}'.format(loc_num)][
                            'datacapture_'+ch]['echogram_'+ch].attrs[gps_cluster_str]) == str:
                            gps_data = dset['location_{:d}'.format(loc_num)][
                                'datacapture_'+ch]['echogram_'+ch].attrs[gps_cluster_str]
                        else:
                            gps_data = dset['location_{:d}'.format(loc_num)][
                                'datacapture_'+ch]['echogram_'+ch].attrs[gps_cluster_str].decode('utf-8')
                        #obtain recorded values from xml file (GPS cluster) and assign to temporary x,y,z,t
                        if (float(_xmlGetVal(gps_data, gps_fix_str)) > 0) and (
                                float(_xmlGetVal(gps_data, gps_message_str)) > 0):
                            try:
                                elev[loc_num] = float(_xmlGetVal(gps_data, alt_asl))
                                lat[loc_num] = float(_xmlGetVal(gps_data, 'Lat'))
                                lon[loc_num] = float(_xmlGetVal(gps_data, 'Long'))
                                d[loc_num], m[loc_num], yr[loc_num], h[loc_num], mi[loc_num], s[loc_num] = _dt_from_comment(echogram)
                            except:          
                                elev[loc_num] = np.nan
                                lat[loc_num] = np.nan
                                lon[loc_num] = np.nan
                                d[loc_num] = np.nan
                        else:
                            elev[loc_num] = np.nan
                            lat[loc_num] = np.nan
                            lon[loc_num] = np.nan
                            d[loc_num] = np.nan 
                    #Append results for each attribute for a line to a final array 
                    x = np.append(x,_dm2dec(lon))
                    y = np.append(y,_dm2dec(lat))
                    z = np.append(z, elev)
                    L_name = np.append(L_name, np.full((tnum,),dset_name))
                    F_name = np.append(F_name, np.full((tnum,),str(file)))
                    day = np.append(day, d)
                    month = np.append(month, m)
                    year = np.append(year, yr)
                    hour = np.append(hour, h)
                    minute = np.append(minute, mi)
                    seconds = np.append(seconds, s)    
        else:
            print(file + 'is not supported. Only .hdf5 are accepted.')
    #Format database 
    final = {'File Name': F_name,
             'Line Name': L_name,
             'Day': day,
             'Month': month,
             'Year': year,
             'X - lon': x,
             'Y - lat': y,
             'Z - Elevation': z,
             'Hour': hour,
             'Minute': minute,
             'Seconds': seconds}
    df = pd.DataFrame(data = final, index = None)
    #Save to excel for convinience 
    if Excel:
        print('Saving dataframe to excel')
        df.to_excel('Radar_GPSdata.xlsx')          
    return df