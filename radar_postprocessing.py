# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 17:17:03 2022

@author: trunzc
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import glaciofunc as gf
from scipy.interpolate import RectBivariateSpline 

def get_data_from_profile(line, data, layer_name):    
    rbs_surface = RectBivariateSpline(data['easting'], data['northing'][::-1], data[layer_name][::-1].T)   
    return rbs_surface.ev(line.easting, line.northing)

#%% Load field gps data

# GPS data from the radar instrument. exported by Kirill
radar = pd.read_csv('G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/Radar/Radar_GPS_lake_europa_2022.csv',
                    index_col='datetime',
                    parse_dates=True)
# index1 = radar.easting.all()>-500000 or radar.easting.all()<-540000
# index2 = 
radar = radar[radar.easting<-500000]
radar = radar[radar.easting>-540000]
# radar = radar.drop(index2.index)

# GPS data from the precision GPS. processed by Celia
gps_rover = pd.read_csv('G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/EuropaGPSdata/Precision_GPS_rover_lake_europa_2022.csv',
                   index_col='time_utc',
                   parse_dates=True)

# GPS data fromt the hand held gps. comes from the combining of Christian and Georgia's GPS
gps_hh = pd.read_csv('G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/EuropaGPSdata/HandHeld_GPS_rover_lake_europa_2022.csv',
                     index_col='time',
                     parse_dates=True)

#%% load elevation data
bedmachine = gf.get_netcdf_data('J:/QGreenland_v2.0.0/Additional/BedMachineGreenland_V5/BedMachineGreenland-v5.nc', 
                             data_type='Bedmachine')

# arcticdem_2m = gf.get_geotiff_data('J:/QGreenland_v2.0.0/Additional/Arctic DEM/Arctic_DEM_mosaic_merged_2m.tif')
arcticdem_10m = gf.get_geotiff_data('J:/QGreenland_v2.0.0/Additional/Arctic DEM/Arctic_DEM_mosaic_merged_10m.tif')
# arcticdem_32m = gf.get_geotiff_data('J:/QGreenland_v2.0.0/Additional/Arctic DEM/Arctic_DEM_mosaic_merged_32m.tif')

#%%
plt.figure()

date_range = [pd.to_datetime('2022-08-11 00:00'),pd.to_datetime('2022-08-23 00:00')]


gps_rover['ellipsoida'].plot(marker='.',linestyle='',label='gps - rover')
gps_hh['ele'].plot(marker='.',linestyle='',label='gps - hand held')
radar['Z - Elevat'].plot(marker='.',linestyle='',label='radar')
plt.xlim(date_range)
plt.ylim(1000,1600)
plt.ylabel('m.a.s.l.')
plt.legend()

#%%
fig,ax = plt.subplots(2)

date_range = [pd.to_datetime('2022-08-11 00:00'),pd.to_datetime('2022-08-23 00:00')]

ax[0].plot(gps_rover.index, gps_rover['easting'], marker='.',linestyle='',label='gps - rover') # - timedelta(hours=24)
ax[0].plot(gps_hh.index, gps_hh['easting'], marker='.',linestyle='',label='gps - hand held')
ax[0].plot(radar.index,radar['easting'],marker='.',linestyle='',label='radar')

ax[0].set_xlim(date_range)
ax[0].set_ylim(-525000,-510000)
ax[0].set_ylabel('m.a.s.l.')
ax[0].legend()

ax[1].plot(gps_rover.index, gps_rover['easting'], marker='.',linestyle='',label='gps - rover') # - timedelta(hours=24)
ax[1].plot(gps_hh.index, gps_hh['easting'], marker='.',linestyle='',label='gps - hand held')
ax[1].plot(radar.index,radar['easting'],marker='.',linestyle='',label='radar')

ax[1].set_xlim(date_range)
ax[1].set_ylim(-525000,-510000)
ax[1].set_ylabel('m.a.s.l.')
ax[1].legend()
# difference between bedmachine and gps data



#%%

radar['z_bedmachine'] = get_data_from_profile(radar, bedmachine, 'surface')
radar['z_arcticdem'] = get_data_from_profile(radar, arcticdem_10m, 'grid_interpolated')


#diff_radar_bedmachine
#diff_radar_arcticdem

gps_rover['z_bedmachine']= get_data_from_profile(gps_rover, bedmachine, 'surface')
gps_rover['z_arcticdem'] = get_data_from_profile(gps_rover, arcticdem_10m, 'grid_interpolated')

#diff_gps_rover_bedmachine

gps_hh['z_bedmachine'] = get_data_from_profile(gps_hh, bedmachine, 'surface')
gps_hh['z_arcticdem'] = get_data_from_profile(gps_hh, arcticdem_10m, 'grid_interpolated')

radar.to_csv('G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/Radar/radar_with_z.csv')
#%%

date_range = [pd.to_datetime('2022-08-11 00:00'),pd.to_datetime('2022-08-23 00:00')]

fig,ax = plt.subplots(3, sharex=True, sharey=True, figsize=(20,10))
fig.suptitle('Elevation comparison', fontsize=16)

ax[0].set_title('Radar GPS')
ax[0].plot(radar.index, radar['z_bedmachine'], marker='.', linestyle='', label='bedmachine')
ax[0].plot(radar.index, radar['z_arcticdem'], marker='.', linestyle='', label='arcticdem')
ax[0].plot(radar.index, radar['Z - Elevat'], marker='.', linestyle='', label='radar', color='black')

ax[1].set_title('Hand Held GPS')
ax[1].plot(gps_hh.index, gps_hh['z_bedmachine'], marker='.', linestyle='', label='bedmachine')
ax[1].plot(gps_hh.index, gps_hh['z_arcticdem'], marker='.', linestyle='', label='arcticdem')
ax[1].plot(gps_hh.index, gps_hh['ele'], marker='.', linestyle='', label='hand held gps', color='black')

ax[2].set_title('Precision GPS')
ax[2].plot(gps_rover.index, gps_rover['z_bedmachine'], marker='.', linestyle='', label='bedmachine')
ax[2].plot(gps_rover.index, gps_rover['z_arcticdem'], marker='.', linestyle='', label='arcticdem')
ax[2].plot(gps_rover.index, gps_rover['ellipsoida'], marker='.', linestyle='', label='precision gps rover', color='black')

ax[2].set_xlim(date_range)
ax[2].set_ylim(1250,1450)

for i in [0,1,2]:
    ax[i].legend()
    ax[i].set_ylabel('(m.a.s.l)')
    
    
#%%

fig,ax = plt.subplots()

ax.scatter(gps_rover.ellipsoida,gps_rover.z_arcticdem)
ax.plot([1250,1450],[1250,1450], color='black')
ax.set_xlim(1250,1450)
ax.set_ylim(1250,1450)
ax.set_ylabel('z arctic dem (m.a.s.l)')
ax.set_xlabel('z precision gps rover (m.a.s.l)')

#%%

fig,ax = plt.subplots()

ax.scatter
x = gps_rover.easting
y = gps_rover.northing
z = gps_rover.ellipsoida - gps_rover.z_arcticdem
colors = plt.cm.PuRd(np.arange(len(z)))

sns.scatterplot(x,y,data=z, hue=z)


























