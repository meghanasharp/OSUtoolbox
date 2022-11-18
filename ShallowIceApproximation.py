# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 11:26:39 2022

@author: trunzc
"""



import pandas as pd
import matplotlib.pyplot as plt
#from matplotlib.pyplot import cm
import numpy as np

#load profile data

profile = pd.read_csv('profile_data_lake_europa.csv', index_col=0)#, index_col='distance').iloc[:, 1:]
# 'easting', 'northing', 
# 'ramco_2m_air_temperature_kelvin',
# 'ramco_total_precipitation_mm_water_equivalent',
# 'ramco_runoff_mm_water_equivalent',
# 'ramco_snowmelt_mm_water_equivalent',
# 'ramco_sublimation_mm_water_equivalent',
# 'bedmachine_ice_surface_elevation_masl', 
# 'bedmachine_ice_thickness_m',
# 'bedmachine_bed_elevation_masl', 
# 'promice_velocity_mperday'

#find lake extent on profile:
    
lake_extent_sw = [-521139.211,-1194232.358]
lake_extent_ne = [-514887.923,-1190764.258]

radar_extent_sw = [-522520.647,-1194383.272]
radar_extent_ne = [-512374.638,-1190389.877]

def find_extent_on_profile(extent_lower_left_corner, extent_upper_right_corner):
    area = profile.distance[(profile.easting > extent_lower_left_corner[0]) & 
                                 (profile.easting < extent_upper_right_corner[0]) & 
                                 (profile.northing > extent_lower_left_corner[1]) & 
                                 (profile.northing < extent_upper_right_corner[1]) ]
    
    return area.reset_index(drop=True)
    
lake_area = find_extent_on_profile(lake_extent_sw, lake_extent_ne)    

radar_area = find_extent_on_profile(radar_extent_sw, radar_extent_ne)   

#%%
#Shallow ice approximation SIA
#Cuffey and Patterson, Chapter 8, equation 8.35




n = 3 #from Cuffey and Patterson
ub = 0 # if we assume no sliding???
rho_ice = 917 #kg/m3
g = 9.81 #m/s2 (N/kg)

#creep paramter
#exemple values page 72-74 in Cuffey and Patterson Chap 3.4.6
#A = 3.5 * 10**(-25) #s-1Pa-3

cuffey = pd.read_csv('A_values_cuffeyPatterson_table3_4.csv')

#ice thickness
H = profile.bedmachine_ice_thickness_m.values #m
surface_slope = profile.arcticdem_10m_ice_surface_elevation_masl.values #profile.bedmachine_ice_surface_elevation_masl.values

x = profile.distance.values #m

def calculate_alpha(y,x):
    y = np.insert(y, 1, y[0])#.insert[-1,profile.bedmachine_ice_thickness_m[-1])
    y = np.append(y, y[-1])
    dy = y[2:]-y[:-2]
    x = np.insert(x, 1, x[0])#.insert[-1,profile.bedmachine_ice_thickness_m[-1])
    x = np.append(x, x[-1])
    dx = x[2:]-x[:-2]
    alpha = dy/dx
    return -alpha

alpha_raw = calculate_alpha(surface_slope,x)

df = pd.DataFrame({'alpha':alpha_raw})

alpha = df.alpha.rolling(window=60,center=True).mean()

# #smooth surface slope
# kernel_size = 20
# array = np.arange(len(kernel))
# kernel = np.ones(kernel_size) / kernel_size
# alpha = np.convolve(alpha_raw, kernel, mode='same')



#%%

#slope
#!!! beware of numberical instability caused by a slope to large compare to delta x, 
# and then if delta x is small, there can be numerical instability with delta t
# 1. always take the smaller timestep between delta slope and delta H
#2. always take a timestep that will create a small enough ice motion compare to delta x

number = 3
fig,ax = plt.subplots(number, sharex=True)

ax[0].plot(profile.distance, profile.arcticdem_10m_ice_surface_elevation_masl,label='ice surface elevation - arctic dem 32m')
ax[0].fill_between(profile.distance, profile.arcticdem_10m_ice_surface_elevation_masl, profile.bedmachine_bed_elevation_masl, alpha=0.3)
ax[0].plot(profile.distance, profile.bedmachine_bed_elevation_masl, label='bed elevation - bedmachine')
ax[0].axvspan(lake_area[0],lake_area[len(lake_area)-1], alpha=0.3, color='cyan', label='minimum lake extent')
ax[0].axvspan(radar_area[0],radar_area[len(radar_area)-1], alpha=0.3, color='grey', label='radar extent')
ax[0].set_ylabel('(m.a.s.l.)')

ax[1].plot(profile.distance,profile.promice_velocity_mperday*365, label='promice velocity', color='purple', linestyle='--')

color=plt.cm.PuRd(np.linspace(0,1,len(cuffey.A)))


for i,A in enumerate(cuffey.A):
#
    taub = rho_ice *g*H*alpha
    temperature = cuffey['T'][i]
    
    surface_velocity =  (ub + 2*A/(n+1) *taub**n *H)*3600*24 
    #column[i]='T='+str(cuffey['T'])
    #us_approx= H**4 *alpha**3#*3600*24

    ax[1].plot(profile.distance,surface_velocity*365, label='A (%s°C)'%temperature, color=color[i])  
    
#ax[1].plot(profile.distance,us_approx, label='aproximated velocity', color='pink')  
ax[1].axvspan(lake_area[0],lake_area[len(lake_area)-1], alpha=0.3, color='cyan')
ax[1].axvspan(radar_area[0],radar_area[len(radar_area)-1], alpha=0.3, color='grey')
ax[1].set_ylim(-1,50)
ax[1].set_ylabel('(m/year)')


ax[2].plot(profile.distance, alpha_raw, label='slope')
ax[2].plot(profile.distance, alpha, label='smoothed slope (window 600m)')
ax[2].axvspan(lake_area[0],lake_area[len(lake_area)-1], alpha=0.3, color='cyan')
ax[2].axvspan(radar_area[0],radar_area[len(radar_area)-1], alpha=0.3, color='grey')
ax[2].set_ylabel('(-)')

ax[2].set_xlabel('Distance from ice divide (m)')

for i in np.arange(number):
    ax[i].legend(prop={'size': 6})


