# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 15:19:27 2023

@author: trunzc
"""




import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
#import glaciofunc as gf
#from scipy.interpolate import RectBivariateSpline,griddata
from matplotlib import colors
import datetime



    
    

##############################################################################
# Load field gps data
##############################################################################


# previous lake position
lake = pd.read_csv('G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/Previous_dataset/naqlk_2014_water_detected.csv')

# GPS data from the radar instrument. exported by Kirill
radar = pd.read_csv('G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/Radar/radar_GPS_datetime_stereographic_2dinterp.csv',
                    index_col='datetime',
                    parse_dates=True) 

# GPS data from the precision GPS. processed by Celia
gps_rover = pd.read_csv('G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/GPSdata_LakeEuropa/Precision_GPS_rover_lake_europa_2022.csv',
                   index_col='time_utc',
                   parse_dates=True)


# GPS data fromt the hand held gps. comes from the combining of Christian and Georgia's GPS
gps_hh = pd.read_csv('G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/GPSdata_LakeEuropa/HandHeld_GPS_rover_lake_europa_2022.csv',
                     index_col='time',
                     parse_dates=True)




##############################################################################
# remove hiawatha data
##############################################################################

radar = radar[radar.easting<-500000]
radar = radar[radar.easting>-540000]

gps_rover = gps_rover[gps_rover.easting<-500000]
gps_rover = gps_rover[gps_rover.easting>-540000]

gps_hh = gps_hh[gps_hh.easting<-500000]
gps_hh = gps_hh[gps_hh.easting>-540000]

##############################################################################
#%% fix radar time index shift
##############################################################################
#the timezone of the data is unknown. the radar time is 19h behind the GPS time
#and the radar is a few meter behind the GPS, so we substract 3.5 min, 
#which seems to best fit the time it takes the radar to be at the same position as the GPS based on easting position
#radar['index_original'] = radar.index.copy()
#radar['index_corrected'] = radar.index.copy()
#radar.index = radar.index + timedelta(hours=19) - timedelta(minutes=3.5)


#%% KI --- re-indexing each line separatly
table = pd.read_csv('G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/Radar/time_shift_for_lines.csv')
#change path for G drive
radar['shift'] = ''
for i,timeshift in enumerate(table.time_shift):
    selection = (radar['File Name'] == table.file_name[i]) & (radar['Line Name'] == table.line_name[i])
    radar['shift'][selection] = radar[selection].index + timedelta(hours=timeshift) - timedelta(minutes=3.5)
radar = radar.set_index('shift')



# table = pd.read_csv('G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/Radar/time_shift_for_lines.csv')
# #change path for G drive
# #radar.index = radar.index.to_series() #switch back to pre-shifted times
# #radar['index_original'] = radar.index

# #file_names = radar['File Name'].unique()

# for i,timeshift in enumerate(table.time_shift):
#     selection = (radar['File Name'] == table.file_name[i]) & (radar['Line Name'] == table.line_name[i])
#     print(any(selection==True))
#     radar = radar[selection].index + timedelta(hours=timeshift)


#     #radar[selection].index = radar[selection].index + timedelta(hours=timeshift)
#     #radar.loc[selection,'index'] = radar[selection].index_original+ timedelta(hours=timeshift)
#     #radar[selection].index_corrected = radar[selection].index_corrected + timedelta(hours=timeshift)
    

# for file_name in file_names:
#     index = radar['File Name'] == file_name
#     #print('index=' , index)
#     line_names = radar[index]['Line Name'].unique()
#     for line_name in line_names:
#         mask = (time_shift.file_name == file_name)&(time_shift.line_name == line_name)
#         shift = int(time_shift[mask].time_shift)
#         sub_index = (radar['File Name'] == file_name)&(radar['Line Name'] == line_name)
#         print('sub_index =', sub_index)
#         radar[sub_index].index = radar[sub_index].index + pd.Timedelta(hours=shift) - pd.Timedelta(minutes=3.5)


##############################################################################
#%% interpolate z radar from GPS data
##############################################################################

#create empty column to insert the provenance of the z data
radar['z_interp_origin'] = ''

# bolean index for the 1d interpolation where no rover data is available
selection_1 = radar.index > pd.to_datetime('2022-08-21 20:00')

#interpolate data along the radar timeserie using the rover gps z data
radar['z_interp'] = np.interp(radar.index,
                        gps_rover.index,
                        gps_rover.ellipsoida)

#remove interpolation values where not rover data is available
radar.z_interp[selection_1] = radar.z_arcticdem[selection_1].values
radar['z_interp_origin'][selection_1]='Arctic DEM 10m'
radar['z_interp_origin'][~selection_1]='GPS rover'

#add arctic dem interpolation values for missing radar positions
#radar.z_interp[selection_1] =



fig,ax = plt.subplots(figsize=(40,15))

#
date_range = [pd.to_datetime('2022-08-13 00:00'),pd.to_datetime('2022-08-25 00:00')]

#ax.axvline(radar.index[radar.z_arcticdem.notna()].values, color = 'black')
ax.plot(radar.index, radar.z_arcticdem, marker='o',linestyle='',label='Arctic DEM',color='black')
ax.plot(gps_rover.index, gps_rover['ellipsoida'], marker='o',linestyle='',label='GPS rover',color='grey')

ax.plot(radar.index ,radar['Z - Elevat'], marker='.',linestyle='',label='radar', color='cyan')

ax.plot(radar.index[radar['z_interp_origin']=='GPS rover'] ,
        radar.z_interp[radar['z_interp_origin']=='GPS rover'],
        marker='.',linestyle='',label='interpolated with rover', color='orangered')
ax.plot(radar.index[radar['z_interp_origin']=='Arctic DEM 10m'] ,
        radar.z_interp[radar['z_interp_origin']=='Arctic DEM 10m'],
        marker='.',linestyle='',label='interpolated with Arctic DEM', color='orange')


ax.set_xlim(date_range)
#ax[0].set_ylim(-525000,-510000)
ax.set_ylabel('m.a.s.l')
ax.legend()




