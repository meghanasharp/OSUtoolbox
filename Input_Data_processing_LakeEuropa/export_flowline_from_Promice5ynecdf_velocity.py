# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 14:32:35 2022

@author: trunzc
"""
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
import netCDF4 as nc
import seaborn as sns
import pandas as pd
from math import radians, cos, sin, asin, sqrt
#from scipy.spatial import KDTree
from scipy.interpolate import RectBivariateSpline    # interpolate grid data to point coordinates


#%% import data

#Velocity netcdf file from 
#Solgaard, A. et al. Greenland ice velocity maps from the PROMICE project. Earth System Science Data 13, 3491–3512 (2021).
#Downloaded in the Dataverse on this page: https://dataverse.geus.dk/dataverse/Ice_velocity/
#"Multi-year Ice Velocity Mosaics for the Greenland Ice Sheet from Sentinel-1 Edition 1"

#path = 'D:/Dropbox/RESEARCH/Qgreenland_LakeEuropa/QGreenland_v2.0.0/Glaciology\Ice sheet velocity/PROMICE Multi-year ice velocity/'
#filename = 'Promice_AVG5year.nc'

ds = nc.Dataset('D:/Dropbox/RESEARCH/Qgreenland_LakeEuropa/QGreenland_v2.0.0/Glaciology\Ice sheet velocity/PROMICE Multi-year ice velocity/Promice_AVG5year.nc')
print(ds.variables.keys())

# condition_x = (ds['x'][:]>-1e6) & (ds['x'][:]<1.5e6)
# condition_y = (ds['y'][:]>-6e5) & (ds['y'][:]<0)

#initial coordinates of the flowlines
#extracted from Qgis, using the CWGC_trackpoints - is it well georeferenced? Kirill had issues with projection
#coordinate system and projection: WGS 84 / NSIDC Sea Ice Polar Stereographic North EPSG:3413
camp = [-516335.771, -1191434.196] #x and y


#%% extract x and y velocites arrays from netcdf

velocity_easting = ds['land_ice_surface_easting_velocity'][:][0]#[condition_x,condition_y]
velocity_northing = ds['land_ice_surface_northing_velocity'][:][0]#[condition_x,condition_y]
velocity_magnitude = ds['land_ice_surface_velocity_magnitude'][:][0]#[condition_x,condition_y]
longitude = columns=ds['x'][:]#[condition_x]
latitude = columns=ds['y'][:]#[condition_y]

# #create dataframe and plot all data

# #create dataframe with x and y velocity components.
# df_velocity_easting = pd.DataFrame(velocity_easting,
#                                 latitude, 
#                                 longitude)

# df_velocity_northing = pd.DataFrame(velocity_northing,
#                                  latitude, 
#                                  longitude)
# plt.figure()
# sns.heatmap(velocity_easting,vmin=-0.5, vmax=0.5)

#%% 

# replace nan with -9999 if there is any
# velocity_easting[np.isnan(velocity_easting)] = -9999
# velocity_northing[np.isnan(velocity_northing)] = -9999
# velocity_magnitude[np.isnan(velocity_magnitude)] = -9999

# use rbs to create a special matrice to enable position searching
rbs_easting = RectBivariateSpline(longitude, latitude[::-1], velocity_easting[::-1].T)# np.fliplr(velocity_easting)) # very fast interpolation !!
rbs_northing = RectBivariateSpline(longitude, latitude[::-1], velocity_northing[::-1].T)#np.fliplr(velocity_northing)) # very fast interpolation !!
rbs_magnitude = RectBivariateSpline(longitude, latitude[::-1], velocity_magnitude[::-1].T)#np.fliplr(velocity_northing)) # very fast interpolation !!

#extract initial position of the flow line
position_easting = camp[0]
position_northing = camp[1]

#initiallize lists
flowline_easting = []
flowline_northing = []
velocity_profile_easting = []
velocity_profile_northing = []
velocity_profile_amplitude = []
velocity_profile_magnitude = []


#loop through the flowline position finding
#DOWNSTREAM
for time in range(3450):
    #record how many years it runs
    time_downstream = time+1
    #pull out the vx,vy components in the velocity fields in m/y for a specific point
    local_velocity_easting = rbs_easting.ev(position_easting, position_northing)*365
    local_velocity_northing = rbs_northing.ev(position_easting, position_northing)*365
    local_velocity_magnitude = rbs_magnitude.ev(position_easting, position_northing)*365#*30.4375
    #calculate position downstream for next timestep based on the velocity of the ice
    position_easting = position_easting+local_velocity_easting
    position_northing = position_northing+local_velocity_northing
    velocity_profile_amplitude.append(np.sqrt(local_velocity_easting**2+local_velocity_northing**2))

    if velocity_profile_amplitude[-1] >= 0:
        #append downstream flowline position and velocity at the end of the list
        flowline_easting.append(position_easting) 
        flowline_northing.append(position_northing)

        velocity_profile_easting.append(local_velocity_easting)
        velocity_profile_northing.append(local_velocity_northing)
        velocity_profile_magnitude.append(local_velocity_magnitude)
    else: 
        break
 
#extract initial position of the flow line
position_easting = camp[0]
position_northing = camp[1]    

#UPSTREAM
for time in range(3500):
    #record how many years it runs
    time_upstream = time+1
    #pull out the vx,vy components in the velocity fields in m/y for a specific point
    local_velocity_easting = rbs_easting.ev(position_easting, position_northing)*365 #*30.4375
    local_velocity_northing = rbs_northing.ev(position_easting, position_northing)*365#*30.4375
    local_velocity_magnitude = rbs_magnitude.ev(position_easting, position_northing)*365#*30.4375
    #calculate position downstream for next timestep based on the velocity of the ice
    position_easting = position_easting-local_velocity_easting
    position_northing = position_northing-local_velocity_northing
    velocity_profile_amplitude.insert(0,np.sqrt(local_velocity_easting**2+local_velocity_northing**2))
    
    if velocity_profile_amplitude[-1] >= 0.5:    
    #calculate position upstream for next timestep based on the velocity of the ice
        #append upstream flowline position and velocity at the beginning of the list
        flowline_easting.insert(0,position_easting) 
        flowline_northing.insert(0,position_northing)

        velocity_profile_easting.insert(0,local_velocity_easting)
        velocity_profile_northing.insert(0,local_velocity_northing)    
        velocity_profile_magnitude.insert(0,local_velocity_magnitude)
    else: 
        break 

    # if np.size(local_velocity_easting)>2:
    #     if np.sign(velocity_profile_easting[-2])*np.sign(velocity_profile_easting[-1])<0 or np.sign(velocity_profile_northing[-2])*np.sign(velocity_profile_northing[-1])<0:
    #         # #remove last value from array
    #         # flowline_easting = flowline_easting[:-1]
    #         # flowline_northing = flowline_northing[:-1]
    #         # velocity_profile_easting = velocity_profile_easting[:-1]
    #         # velocity_profile_northing = velocity_profile_northing[:-1]  
    #         break                                                   
       
    #if oppositeSigns(velocity_profile_easting[-1], local_velocity_easting) == True or oppositeSigns(velocity_profile_northing[-1], local_velocity_northing) == True:
    #    break
    #print(Year_Vxs)
    # Year_Vys = rbs_northing.ev(KinemGPS_Easts_migrated, KinemGPS_Norths_migrated) / 365.0
    
profile = np.cumsum(velocity_profile_magnitude)
#%%
extent = [-650250, 849750, -3349750, -649750]
fig, ax = plt.subplots()
#plot velocity map 
ax.imshow(velocity_magnitude,extent=extent,vmin=0,vmax=0.05)
#plot flow line
sc = ax.scatter(flowline_easting,flowline_northing,c=range(time_upstream+time_downstream),cmap=cm.jet)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('years')
#ax.plot(flowline_easting,flowline_northing, color='black')
ax.set_xlim([-5.6e5,-4.9e5])
ax.set_ylim([-1.26e6,-1.18e6])

#%%

fig,ax = plt.subplots(3)
ax[0].plot(profile/1000, velocity_profile_magnitude, label='magnitude')
ax[0].plot(profile/1000, velocity_profile_amplitude, label='calculated amplitude')
ax[0].legend()

ax[1].plot(profile/1000, velocity_profile_easting, label='velocity easting')
ax[1].legend()

ax[2].plot(profile/1000, velocity_profile_northing, label='velocity northing')
ax[2].set_xlabel('Distance from ice divide (kn)')
ax[2].legend()


#%% inpterpolate to get a equidistant profile points

from scipy.interpolate import interp1d
import numpy as np

x = np.copy(flowline_easting)
y = np.copy(flowline_northing)

# Linear length on the line
distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ))
distance_total = distance[-1]
distance = distance/distance_total

fx, fy = interp1d( distance, x ), interp1d( distance, y )

alpha = np.linspace(0, int(distance[-1]), 350) #or use arange???
flowline_interp_easting, flowline_interp_northing = fx(alpha), fy(alpha)

print(distance_total/350, ' m')

plt.figure()
plt.plot(x, y, '-')
plt.plot(flowline_interp_easting, flowline_interp_northing, 'or')
plt.axis('equal')

df = pd.DataFrame({'longitude':flowline_interp_easting,
                  'latitude':flowline_interp_northing
                  })


df.to_csv('europa_flowline_coordinates_110m.csv')
























