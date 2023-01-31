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
from scipy.interpolate import RectBivariateSpline,griddata
from matplotlib import colors

def get_data_from_profile(line, data, layer_name):    
    rbs_surface = RectBivariateSpline(data['easting'], data['northing'][::-1], data[layer_name][::-1].T)   
    return rbs_surface.ev(line.easting, line.northing)


def plot_2D(x,y,z,
            title='z arctic dem minus z precision gps',
            xlabel='Easting (m)',
            ylabel='Northing (m)',
            units='(m)'):

    fig,ax = plt.subplots(figsize=(10,10))
    
    cmap = plt.cm.rainbow
    #norm = colors.BoundaryNorm(np.arange(round(min(z)),round(max(z)),10), cmap.N)
    norm = colors.BoundaryNorm(np.arange(-1,1,0.1), cmap.N)
    #for i in np. arange(len(lake))
    plt.title(title)
    plt.plot([lake.x_zero,lake.x_end],[lake.y_zero,lake.y_end], color='black')
    plt.scatter(x, y, c=z, cmap=cmap, norm=norm, s=100, edgecolor='none')
    plt.colorbar(ticks=np.arange(-1,1,0.2),label=units,fraction=0.046, pad=0.04)
    plt.axis('square')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    

##############################################################################
# Load field gps data
##############################################################################


# previous lake position
lake = pd.read_csv('G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/Previous_dataset/naqlk_2014_water_detected.csv')

# GPS data from the radar instrument. exported by Kirill
radar = pd.read_csv('G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/Radar/Radar_GPS_lake_europa_2022.csv',
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
# fix radar time index shift
##############################################################################
#the timezone of the data is unknown. the radar time is 19h behind the GPS time
#and the radar is a few meter behind the GPS, so we substract 3.5 min, 
#which seems to best fit the time it takes the radar to be at the same position as the GPS based on easting position

radar.index = radar.index + timedelta(hours=19) - timedelta(minutes=3.5)


# plot to Compare GPS and RADAR values in function of time
fig,ax = plt.subplots(3, sharex=True)

date_range = [pd.to_datetime('2022-08-11 00:00'),pd.to_datetime('2022-08-23 00:00')]

ax[0].plot(gps_rover.index, gps_rover['easting'], marker='.',linestyle='',label='gps - rover') # - timedelta(hours=24)
ax[0].plot(gps_hh.index, gps_hh['easting'], marker='.',linestyle='',label='gps - hand held')
ax[0].plot(radar.index ,radar['easting'],marker='.',linestyle='',label='radar')

ax[0].set_xlim(date_range)
ax[0].set_ylim(-525000,-510000)
ax[0].set_ylabel('Easting (m)')
ax[0].legend()

ax[1].plot(gps_rover.index, gps_rover['northing'], marker='.',linestyle='',label='gps - rover') # - timedelta(hours=24)
ax[1].plot(gps_hh.index, gps_hh['northing'], marker='.',linestyle='',label='gps - hand held')
ax[1].plot(radar.index ,radar['northing'],marker='.',linestyle='',label='radar')

ax[1].set_xlim(date_range)
ax[1].set_ylim(-1198000,-1180000)
ax[1].set_ylabel('Northing (m)')
ax[1].legend()
# difference between bedmachine and gps data

ax[2].plot(gps_rover.index, gps_rover['ellipsoida'], marker='.',linestyle='',label='gps - rover') # - timedelta(hours=24)
ax[2].plot(gps_hh.index, gps_hh['ele'], marker='.',linestyle='',label='gps - hand held')
ax[2].plot(radar.index ,radar['Z - Elevat'],marker='.',linestyle='',label='radar')

ax[2].set_xlim(date_range)
#ax[2].set_ylim(-525000,-510000)
ax[2].set_ylabel('m.a.s.l.')
ax[2].legend()



##############################################################################
# load DEM data
##############################################################################
#uncomment to load the right dataset. 2m dem from arctic dem is pretty heavy

# bedmachine = gf.get_netcdf_data('J:/QGreenland_v2.0.0/Additional/BedMachineGreenland_V5/BedMachineGreenland-v5.nc', data_type='Bedmachine')

# arcticdem_2m = gf.get_geotiff_data('J:/QGreenland_v2.0.0/Additional/Arctic DEM/Arctic_DEM_mosaic_merged_2m.tif')
arcticdem_10m = gf.get_geotiff_data('J:/QGreenland_v2.0.0/Additional/Arctic DEM/Arctic_DEM_mosaic_merged_10m.tif')
# arcticdem_32m = gf.get_geotiff_data('J:/QGreenland_v2.0.0/Additional/Arctic DEM/Arctic_DEM_mosaic_merged_32m.tif')


##############################################################################
##############################################################################
#%% DATA PROCESSING
##############################################################################
##############################################################################


##############################################################################
# Extract z position from bedmachine and arctic dem
##############################################################################



# #['z_bedmachine'] = get_data_from_profile(radar, bedmachine, 'surface')
radar['z_arcticdem'] = get_data_from_profile(radar, arcticdem_10m, 'grid_interpolated')


#diff_radar_bedmachine
#diff_radar_arcticdem

#gps_rover['z_bedmachine']= get_data_from_profile(gps_rover, bedmachine, 'surface')
gps_rover['z_arcticdem'] = get_data_from_profile(gps_rover, arcticdem_10m, 'grid_interpolated')
gps_rover['z_diff_arcticdem'] = gps_rover.z_arcticdem - gps_rover.ellipsoida


#save data to csv
gps_rover.to_csv('gps_rover_vs_arcticdem.csv')


# #diff_gps_rover_bedmachine

# gps_hh['z_bedmachine'] = get_data_from_profile(gps_hh, bedmachine, 'surface')
gps_hh['z_arcticdem'] = get_data_from_profile(gps_hh, arcticdem_10m, 'grid_interpolated')

# radar.to_csv('G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/Radar/radar_with_z.csv')

# Compare Field data with each other
##############################################################################
date_range = [pd.to_datetime('2022-08-11 00:00'),pd.to_datetime('2022-08-23 00:00')]

fig,ax = plt.subplots(3, sharex=True, sharey=True, figsize=(20,10))
fig.suptitle('Elevation comparison', fontsize=16)

ax[0].set_title('Radar GPS')
#ax[0].plot(radar.index, radar['z_bedmachine'], marker='.', linestyle='', label='bedmachine')
ax[0].plot(radar.index, radar['z_arcticdem'], marker='.', linestyle='', label='arcticdem')
ax[0].plot(radar.index, radar['Z - Elevat'], marker='.', linestyle='', label='radar', color='black')

ax[1].set_title('Hand Held GPS')
#ax[1].plot(gps_hh.index, gps_hh['z_bedmachine'], marker='.', linestyle='', label='bedmachine')
ax[1].plot(gps_hh.index, gps_hh['z_arcticdem'], marker='.', linestyle='', label='arcticdem')
ax[1].plot(gps_hh.index, gps_hh['ele'], marker='.', linestyle='', label='hand held gps', color='black')

ax[2].set_title('Precision GPS')
#ax[2].plot(gps_rover.index, gps_rover['z_bedmachine'], marker='.', linestyle='', label='bedmachine')
ax[2].plot(gps_rover.index, gps_rover['z_arcticdem'], marker='.', linestyle='', label='arcticdem')
ax[2].plot(gps_rover.index, gps_rover['ellipsoida'], marker='.', linestyle='', label='precision gps rover', color='black')

ax[2].set_xlim(date_range)
ax[2].set_ylim(1250,1450)

for i in [0,1,2]:
    ax[i].legend()
    ax[i].set_ylabel('(m.a.s.l)')
    

# Compare GPS ROVER with ArcticDEM
##############################################################################
fig,ax = plt.subplots()

ax.scatter(gps_rover.ellipsoida,gps_rover.z_arcticdem)
ax.plot([1250,1450],[1250,1450], color='black')
ax.set_xlim(1250,1450)
ax.set_ylim(1250,1450)
ax.set_ylabel('z arctic dem (m.a.s.l)')
ax.set_xlabel('z precision gps rover (m.a.s.l)')

#

# plot the difference in 2D
plot_2D(x=gps_rover.easting,
        y=gps_rover.northing,
        z=gps_rover.z_diff_arcticdem)

#plot histogram
plt.figure()
plt.hist(gps_rover.z_diff_arcticdem,80)
plt.ylabel('counts')
plt.xlabel('Diff arctic dem and precision gps (m)')

##############################################################################
#%% interpolate z radar from GPS data
##############################################################################
# this needs to be refined to prevent data interpolation in area with no data


radar['z_gps'] = np.interp(radar.index,
                        gps_rover.index,
                        gps_rover.ellipsoida)

fig,ax = plt.subplots()

date_range = [pd.to_datetime('2022-08-11 00:00'),pd.to_datetime('2022-08-23 00:00')]

ax.plot(radar.index ,radar['Z - Elevat'],marker='.',linestyle='',label='z from radar')
ax.plot(radar.index ,radar['z_gps'],marker='.',linestyle='',label='z from GPS rover')


ax.set_xlim(date_range)
#ax[0].set_ylim(-525000,-510000)
ax.set_ylabel('m.a.s.l')
ax.legend()







    
    




















