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

def get_data_from_profile(line, data, layer_name):    
    rbs_surface = RectBivariateSpline(data['easting'], data['northing'][::-1], data[layer_name][::-1].T)   
    return rbs_surface.ev(line.easting, line.northing)

#%% Load field gps data

# previous lake position

lake = pd.read_csv('G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/Previous_dataset/naqlk_2014_water_detected.csv')

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
gps_rover = gps_rover[gps_rover.easting<-500000]
gps_rover = gps_rover[gps_rover.easting>-540000]

# GPS data fromt the hand held gps. comes from the combining of Christian and Georgia's GPS
gps_hh = pd.read_csv('G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/EuropaGPSdata/HandHeld_GPS_rover_lake_europa_2022.csv',
                     index_col='time',
                     parse_dates=True)
gps_hh = gps_hh[gps_hh.easting<-500000]
gps_hh = gps_hh[gps_hh.easting>-540000]

#%% load elevation data
# bedmachine = gf.get_netcdf_data('J:/QGreenland_v2.0.0/Additional/BedMachineGreenland_V5/BedMachineGreenland-v5.nc', 
#                              data_type='Bedmachine')

# arcticdem_2m = gf.get_geotiff_data('J:/QGreenland_v2.0.0/Additional/Arctic DEM/Arctic_DEM_mosaic_merged_2m.tif')
arcticdem_10m = gf.get_geotiff_data('J:/QGreenland_v2.0.0/Additional/Arctic DEM/Arctic_DEM_mosaic_merged_10m.tif')
# arcticdem_32m = gf.get_geotiff_data('J:/QGreenland_v2.0.0/Additional/Arctic DEM/Arctic_DEM_mosaic_merged_32m.tif')

#%% Extract z position from bedmachine and arctic dem

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
# gps_hh['z_arcticdem'] = get_data_from_profile(gps_hh, arcticdem_10m, 'grid_interpolated')

# radar.to_csv('G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/Radar/radar_with_z.csv')

#%%
# plt.figure()

# date_range = [pd.to_datetime('2022-08-11 00:00'),pd.to_datetime('2022-08-23 00:00')]


# gps_rover['ellipsoida'].plot(marker='.',linestyle='',label='gps - rover')
# gps_hh['ele'].plot(marker='.',linestyle='',label='gps - hand held')
# radar['Z - Elevat'].plot(marker='.',linestyle='',label='radar')
# plt.xlim(date_range)
# plt.ylim(1000,1600)
# plt.ylabel('m.a.s.l.')
# plt.legend()

# #%%
# fig,ax = plt.subplots(2)

# date_range = [pd.to_datetime('2022-08-11 00:00'),pd.to_datetime('2022-08-23 00:00')]

# ax[0].plot(gps_rover.index, gps_rover['easting'], marker='.',linestyle='',label='gps - rover') # - timedelta(hours=24)
# ax[0].plot(gps_hh.index, gps_hh['easting'], marker='.',linestyle='',label='gps - hand held')
# ax[0].plot(radar.index,radar['easting'],marker='.',linestyle='',label='radar')

# ax[0].set_xlim(date_range)
# ax[0].set_ylim(-525000,-510000)
# ax[0].set_ylabel('m.a.s.l.')
# ax[0].legend()

# ax[1].plot(gps_rover.index, gps_rover['easting'], marker='.',linestyle='',label='gps - rover') # - timedelta(hours=24)
# ax[1].plot(gps_hh.index, gps_hh['easting'], marker='.',linestyle='',label='gps - hand held')
# ax[1].plot(radar.index,radar['easting'],marker='.',linestyle='',label='radar')

# ax[1].set_xlim(date_range)
# ax[1].set_ylim(-525000,-510000)
# ax[1].set_ylabel('m.a.s.l.')
# ax[1].legend()
# # difference between bedmachine and gps data




# #%%

# date_range = [pd.to_datetime('2022-08-11 00:00'),pd.to_datetime('2022-08-23 00:00')]

# fig,ax = plt.subplots(3, sharex=True, sharey=True, figsize=(20,10))
# fig.suptitle('Elevation comparison', fontsize=16)

# ax[0].set_title('Radar GPS')
# ax[0].plot(radar.index, radar['z_bedmachine'], marker='.', linestyle='', label='bedmachine')
# ax[0].plot(radar.index, radar['z_arcticdem'], marker='.', linestyle='', label='arcticdem')
# ax[0].plot(radar.index, radar['Z - Elevat'], marker='.', linestyle='', label='radar', color='black')

# ax[1].set_title('Hand Held GPS')
# ax[1].plot(gps_hh.index, gps_hh['z_bedmachine'], marker='.', linestyle='', label='bedmachine')
# ax[1].plot(gps_hh.index, gps_hh['z_arcticdem'], marker='.', linestyle='', label='arcticdem')
# ax[1].plot(gps_hh.index, gps_hh['ele'], marker='.', linestyle='', label='hand held gps', color='black')

# ax[2].set_title('Precision GPS')
# ax[2].plot(gps_rover.index, gps_rover['z_bedmachine'], marker='.', linestyle='', label='bedmachine')
# ax[2].plot(gps_rover.index, gps_rover['z_arcticdem'], marker='.', linestyle='', label='arcticdem')
# ax[2].plot(gps_rover.index, gps_rover['ellipsoida'], marker='.', linestyle='', label='precision gps rover', color='black')

# ax[2].set_xlim(date_range)
# ax[2].set_ylim(1250,1450)

# for i in [0,1,2]:
#     ax[i].legend()
#     ax[i].set_ylabel('(m.a.s.l)')
    
    
#%%

fig,ax = plt.subplots()

ax.scatter(gps_rover.ellipsoida,gps_rover.z_arcticdem)
ax.plot([1250,1450],[1250,1450], color='black')
ax.set_xlim(1250,1450)
ax.set_ylim(1250,1450)
ax.set_ylabel('z arctic dem (m.a.s.l)')
ax.set_xlabel('z precision gps rover (m.a.s.l)')

#%%

fig,ax = plt.subplots(figsize=(10,10))

# ax.scatter

# #colors = plt.cm.PuRd(np.arange(len(z)))

# sns.scatterplot(x=x, 
#                 y=y, 
#                 hue=z,
#                 hue_norm=(-0.1,0,1),
#                 linewidth=0,)#, alpha = 0.2 )


from matplotlib import colors
x = gps_rover.easting
y = gps_rover.northing
z = gps_rover.z_diff_arcticdem

plt.xlabel('X-axis ')
plt.ylabel('Y-axis ')

cmap = plt.cm.rainbow
#norm = colors.BoundaryNorm(np.arange(round(min(z)),round(max(z)),10), cmap.N)
norm = colors.BoundaryNorm(np.arange(-1,1,0.1), cmap.N)
#for i in np. arange(len(lake))
plt.title('z arctic dem minus z precision gps')
plt.plot([lake.x_zero,lake.x_end],[lake.y_zero,lake.y_end], color='black')
plt.scatter(x, y, c=z, cmap=cmap, norm=norm, s=100, edgecolor='none')
plt.colorbar(ticks=np.arange(-1,1,0.2),label='(m)',fraction=0.046, pad=0.04)
plt.axis('square')
plt.xlabel('Easting (m)')
plt.ylabel('Northing (m)')
#plt.legend()

#plt.show()
#%%
plt.figure()
plt.hist(z,80)
plt.ylabel('counts')
plt.xlabel('Diff arctic dem and precision gps (m)')


#%%

# Generate some test data
x = gps_rover.easting
y = gps_rover.northing
z = gps_rover.z_diff_arcticdem



#heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.imshow(z, extent=extent, origin='lower')
plt.show()


#%%

#First we will read in some data 
#The first column of this file is distance east,
#The 2nd column is distance north
#The 3rd column is elevation
#from pandas import read_csv

east = gps_rover.easting
north = gps_rover.northing
elevation = gps_rover.z_diff_arcticdem
#For tricontour() the arguments are
# three 1D arrays that contain our
# x, y, and z values in that order
plt.figure(figsize=(5,5))
plt.tricontour(east,north,elevation,100)
plt.xlabel('East', fontsize=18)
plt.ylabel('North', fontsize=18)
plt.tight_layout()
#triangular contour plots can be customized by 
# using the same techniques as with contour()


#%%
from scipy.interpolate import griddata

xi = gps_rover.easting.values
yi = gps_rover.northing.values
z = gps_rover.z_diff_arcticdem.values
zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear')
plt.contour(xi, yi, zi)

#%%

#It is also possible to take irregular data and use interpolation
# techniques to construct a regular grid of elevations
east = gps_rover.easting
north = gps_rover.northing
elevation = gps_rover.z_diff_arcticdem

#First, we find the ranges of our x and y variables
eastmin = east.min()
eastmax = east.max()
northmin = north.min()
northmax = north.max()

#Then we determine the density of the grid that we want
n_east_grid = 1000 #10 points in the east direction
n_north_grid = 1000 #10 points in the north direction

#Now we can create 1D arrays that contain the locations on our grid
#  These run from min to max for each variable and have the 
#  number of grid points specified above
grid_x,grid_y = np.mgrid[eastmin:eastmax:n_east_grid,northmin:northmax:n_north_grid]
# east_reg = np.linspace(eastmin,eastmax,n_east_grid)
# north_reg = np.linspace(northmin,northmax,n_north_grid)


#Now we can use a function called griddata(), which performs interpolation
# of irregular data
#Arguments are:
# 1. irregular x values
# 2. irregular y values
# 3. elevations (z values) at irregular locations
# 4. a 1D array of the x values of the regular grid
# 5. a 1D array of the y values of the regular grid
# 6. interp='linear' tells it to do linear interpolation
#For griddata(), we need to use .values on all of the pandas data objects
#to extract only the raw data arrays.  griddata() doesn't know how to 
#handle pandas data objects.

points = gps_rover['easting','northing'].to_numpy()
elevation_grid = griddata(points, elevation.values, (grid_x ,grid_y), method='linear')

# Make a contour plot of the regularly gridded data.
plt.figure()
plt.contour(east_reg, north_reg, elevation_grid,20)

#We can control the fineness of the grid by changing the number of points
# on the grid (remember the density variables above).














