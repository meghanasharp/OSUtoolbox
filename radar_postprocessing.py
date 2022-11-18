# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 17:17:03 2022

@author: trunzc
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import glaciofunc as gf



radar = pd.read_csv('G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/Radar/Radar_GPSdata_cleanedup.csv',
                    parse_dates ={'datetime':['date','time']},
                    usecols = ['date','time','File Name', 'Line Name', 'X - lon', 'Y - lat', 'Z - Elevation'],
                    index_col ='datetime')

radar = radar.dropna()
                


gps_rover = pd.read_csv('G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/EuropaGPSdata/Precision_GPS_rover_lake_europa_2022.csv',
                   index_col='time_utc',
                   parse_dates=True)#,
                   # usecols=['time_utc','time_local','latitude_decimal_degree', 'longitude_decimal_degree',
                   #        'ellipsoidal_height_m'] )

gps_hh = pd.read_csv('G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/EuropaGPSdata/HandHeld_GPS_rover_lake_europa_2022.csv',
                     index_col='time',
                     parse_dates=True)


#%%
plt.figure()

date_range = ['2022-08-11 00:00','2022-08-23 00:00']

radar['Z - Elevation'].plot(marker='.',linestyle='',label='radar')
gps_rover['ellipsoida'].plot(marker='.',linestyle='',label='gps - rover')
gps_hh['ele'].plot(marker='.',linestyle='',label='gps - hand held')
plt.xlim(date_range)
plt.ylim(1000,1600)
plt.ylabel('m.a.s.l.')
plt.legend()


plt.figure()

date_range = ['2022-08-11 00:00','2022-08-23 00:00']

#radar['easting'].plot(marker='.',linestyle='',label='radar')
# gps_rover['easting'].plot(marker='.',linestyle='',label='gps - rover')
# gps_hh['easting'].plot(marker='.',linestyle='',label='gps - hand held')

plt.plot(gps_rover.index + datetime.datetime.timedelta(hours=24), gps_rover['easting'], marker='.',linestyle='',label='gps - rover')
gps_hh['easting'].plot(marker='.',linestyle='',label='gps - hand held')

plt.xlim(date_range)
#plt.ylim(1000,1600)
plt.ylabel('m.a.s.l.')
plt.legend()
#ax.plot(gps['ellipsoidal_height_m'][date_range[0]:date_range[1]])

# difference between bedmachine and gps data

#%%
bedmachine = gf.get_netcdf_data('J:/QGreenland_v2.0.0/Additional/BedMachineGreenland_V5/BedMachineGreenland-v5.nc', 
                             data_type='Bedmachine')

arcticdem_2m = gf.get_geotiff_data('J:/QGreenland_v2.0.0/Additional/Arctic DEM/Arctic_DEM_mosaic_merged_2m.tif')
# arcticdem_10m = gf.get_geotiff_data('J:/QGreenland_v2.0.0/Additional/Arctic DEM/Arctic_DEM_mosaic_merged_10m.tif')
# arcticdem_32m = gf.get_geotiff_data('J:/QGreenland_v2.0.0/Additional/Arctic DEM/Arctic_DEM_mosaic_merged_32m.tif')

#%%


def get_data_from_profile(line, data, layer_name):
    
    rbs_surface = RectBivariateSpline(data['easting'], data['northing'][::-1], data[layer_name][::-1].T) 
    profile = rbs_surface.ev(line.easting, line.northing)
    return profile