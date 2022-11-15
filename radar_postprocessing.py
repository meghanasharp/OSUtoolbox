# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 17:17:03 2022

@author: trunzc
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np



radar = pd.read_csv('G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/Radar/Radar_GPSdata_cleanedup.csv',
                    parse_dates ={'datetime':['date','time']},
                    usecols = ['date','time','File Name', 'Line Name', 'X - lon', 'Y - lat', 'Z - Elevation'],
                    index_col ='datetime')

radar = radar.dropna()
                

        


gps = pd.read_csv('G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/EuropaGPSdata/GPS_concat_rover.csv',
                   index_col='time_utc',
                   parse_dates=True,
                   usecols=['time_utc','time_local','latitude_decimal_degree', 'longitude_decimal_degree',
                          'ellipsoidal_height_m'] )



#%%
plt.figure()

date_range = ['2022-08-11 00:00','2022-08-23 00:00']

radar['Z - Elevation'].plot(marker='.',linestyle='',label='radar')
gps['ellipsoidal_height_m'].plot(marker='.',linestyle='',label='gps')
plt.xlim(date_range)
plt.ylim(1000,1600)
plt.ylabel('m.a.s.l.')
plt.legend()
#ax.plot(gps['ellipsoidal_height_m'][date_range[0]:date_range[1]])

# difference between bedmachine and gps data