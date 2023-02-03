# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 17:17:03 2022

@author: trunzc
"""



import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np
#import glaciofunc as gf
#from scipy.interpolate import RectBivariateSpline,griddata
from matplotlib import colors



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
radar = pd.read_csv('G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/Radar/radar_GPS_datetime_stereographic.csv',
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
radar['index_original'] = radar.index
radar.index = radar.index + timedelta(hours=19) - timedelta(minutes=3.5)


#%% plot correction comparison
fig,ax = plt.subplots(3, sharex=True, figsize=(40,20))

date_range = [pd.to_datetime('2022-08-11 00:00'),pd.to_datetime('2022-08-23 00:00')]

ax[0].set_title('EASTING')
ax[0].plot(gps_rover.index, gps_rover['easting'], marker='o',linestyle='',label='gps - rover', color='black')
ax[0].plot(radar.index_original ,radar['easting'],marker='.',linestyle='',label='radar - original', color='orangered' )
ax[0].plot(radar.index ,radar['easting'],marker='.',linestyle='',label='radar - corrected', color='orange')

ax[0].set_xlim(date_range)
ax[0].set_ylim(-525000,-510000)
ax[0].set_ylabel('(m)')
ax[0].legend(fontsize=15)

ax[1].set_title('NORTHING')
ax[1].plot(gps_rover.index, gps_rover['northing'], marker='o',linestyle='',label='gps - rover', color='black')
ax[1].plot(radar.index_original ,radar['northing'],marker='.',linestyle='',label='radar - original', color='orangered')
ax[1].plot(radar.index ,radar['northing'],marker='.',linestyle='',label='radar - corrected', color='orange')

ax[1].set_xlim(date_range)
ax[1].set_ylim(-1198000,-1180000)
ax[1].set_ylabel('(m)')
#ax[1].legend()

ax[2].set_title('ELEVATION')
ax[2].plot(gps_rover.index, gps_rover['ellipsoida'], marker='o',linestyle='',label='gps - rover', color='black')
ax[2].plot(radar.index_original ,radar['Z - Elevat'],marker='.',linestyle='',label='radar - original', color='orangered')
ax[2].plot(radar.index ,radar['Z - Elevat'],marker='.',linestyle='',label='radar - corrected', color='orange')

ax[2].set_xlim(date_range)
#ax[2].set_ylim(-525000,-510000)
ax[2].set_ylabel('(m.a.s.l.)')
#ax[2].legend()

#%% plot to Compare GPS and RADAR values in function of time
fig,ax = plt.subplots(3, sharex=True, figsize=(40,20))

date_range = [pd.to_datetime('2022-08-11 00:00'),pd.to_datetime('2022-08-23 00:00')]

ax[0].set_title('EASTING')
ax[0].plot(gps_rover.index, gps_rover['easting'], marker='o',linestyle='',label='gps - rover', color='black') # - timedelta(hours=24)
ax[0].plot(gps_hh.index, gps_hh['easting'], marker='.',linestyle='',label='gps - hand held', color='grey')
ax[0].plot(radar.index ,radar['easting'],marker='.',linestyle='',label='radar', color='orange')

ax[0].set_xlim(date_range)
ax[0].set_ylim(-525000,-510000)
ax[0].set_ylabel('(m)')
ax[0].legend(fontsize=15)

ax[1].set_title('NORTHING')
ax[1].plot(gps_rover.index, gps_rover['northing'], marker='o',linestyle='',label='gps - rover', color='black') # - timedelta(hours=24)
ax[1].plot(gps_hh.index, gps_hh['northing'], marker='.',linestyle='',label='gps - hand held', color='grey')
ax[1].plot(radar.index ,radar['northing'],marker='.',linestyle='',label='radar', color='orange')

ax[1].set_xlim(date_range)
ax[1].set_ylim(-1198000,-1180000)
ax[1].set_ylabel('(m)')
#ax[1].legend()
# difference between bedmachine and gps data

ax[2].set_title('ELEVATION')
ax[2].plot(gps_rover.index, gps_rover['ellipsoida'], marker='o',linestyle='',label='gps - rover', color='black') # - timedelta(hours=24)
ax[2].plot(gps_hh.index, gps_hh['ele'], marker='.',linestyle='',label='gps - hand held', color='grey')
ax[2].plot(radar.index ,radar['Z - Elevat'],marker='.',linestyle='',label='radar', color='orange')

ax[2].set_xlim(date_range)
#ax[2].set_ylim(-525000,-510000)
ax[2].set_ylabel('(m.a.s.l.)')
#ax[2].legend()


#%% plot by files

file_names = radar['File Name'].unique()

for file_name in file_names:
    index = radar['File Name'] == file_name

    fig,ax = plt.subplots(3, sharex=True, figsize=(40,20))
    fig.suptitle('File %s'%file_name)
    date_range = [pd.to_datetime('2022-08-11 00:00'),pd.to_datetime('2022-08-26 00:00')]
    
    ax[0].set_title('EASTING')
    ax[0].plot(gps_rover.index, gps_rover['easting'], marker='o',linestyle='',label='gps - rover', color='black') # - timedelta(hours=24)
    ax[0].plot(gps_hh.index, gps_hh['easting'], marker='.',linestyle='',label='gps - hand held', color='grey')
    ax[0].plot(radar.index[index] ,radar['easting'][index],marker='.',linestyle='',label='radar', color='orange')
    
    ax[0].set_xlim(date_range)
    ax[0].set_ylim(-525000,-510000)
    ax[0].set_ylabel('(m)')
    ax[0].legend(fontsize=15)
    
    ax[1].set_title('NORTHING')
    ax[1].plot(gps_rover.index, gps_rover['northing'], marker='o',linestyle='',label='gps - rover', color='black') # - timedelta(hours=24)
    ax[1].plot(gps_hh.index, gps_hh['northing'], marker='.',linestyle='',label='gps - hand held', color='grey')
    ax[1].plot(radar.index[index] ,radar['northing'][index],marker='.',linestyle='',label='radar', color='orange')
    
    ax[1].set_xlim(date_range)
    ax[1].set_ylim(-1198000,-1180000)
    ax[1].set_ylabel('(m)')
    #ax[1].legend()
    # difference between bedmachine and gps data
    
    ax[2].set_title('ELEVATION')
    ax[2].plot(gps_rover.index, gps_rover['ellipsoida'], marker='o',linestyle='',label='gps - rover', color='black') # - timedelta(hours=24)
    ax[2].plot(gps_hh.index, gps_hh['ele'], marker='.',linestyle='',label='gps - hand held', color='grey')
    ax[2].plot(radar.index[index] ,radar['Z - Elevat'][index],marker='.',linestyle='',label='radar', color='orange')
    
    ax[2].set_xlim(date_range)
    #ax[2].set_ylim(-525000,-510000)
    ax[2].set_ylabel('(m.a.s.l.)')
    #ax[2].legend()
    #plt.savefig('radar_comparison_Line_%s'%file_name)



# #%% plot by lines
# from matplotlib.pyplot import cm
# file_names = radar['File Name'].unique()


# for file_name in file_names:
#     index = radar['File Name'] == file_name
#     line_names = radar['Line Name'][index].unique()    

#     fig,ax = plt.subplots(3, sharex=True, figsize=(40,20))
#     fig.suptitle('File %s'%file_name)
#     date_range = [pd.to_datetime('2022-08-11 00:00'),pd.to_datetime('2022-08-26 00:00')]
    
#     ax[0].set_title('EASTING')
#     ax[0].plot(gps_rover.index, gps_rover['easting'], marker='o',linestyle='',label='gps - rover', color='black') # - timedelta(hours=24)
#     ax[0].plot(gps_hh.index, gps_hh['easting'], marker='.',linestyle='',label='gps - hand held', color='grey')

    
#     ax[0].set_xlim(date_range)
#     ax[0].set_ylim(-525000,-510000)
#     ax[0].set_ylabel('(m)')
#     ax[0].legend(fontsize=15)
    
#     ax[1].set_title('NORTHING')
#     ax[1].plot(gps_rover.index, gps_rover['northing'], marker='o',linestyle='',label='gps - rover', color='black') # - timedelta(hours=24)
#     ax[1].plot(gps_hh.index, gps_hh['northing'], marker='.',linestyle='',label='gps - hand held', color='grey')

    
#     ax[1].set_xlim(date_range)
#     ax[1].set_ylim(-1198000,-1180000)
#     ax[1].set_ylabel('(m)')
#     #ax[1].legend()
#     # difference between bedmachine and gps data
    
#     ax[2].set_title('ELEVATION')
#     ax[2].plot(gps_rover.index, gps_rover['ellipsoida'], marker='o',linestyle='',label='gps - rover', color='black') # - timedelta(hours=24)
#     ax[2].plot(gps_hh.index, gps_hh['ele'], marker='.',linestyle='',label='gps - hand held', color='grey')


#     color = cm.rainbow(np.linspace(0, 1, len(line_names)))
#     for i,line_name in enumerate(line_names):
#         index_index = radar['Line Name'][index]==line_name
        
#         ax[0].plot(radar.index[index_index] ,radar['easting'][index_index],marker='.',linestyle='',label='radar', color=color[i])    
#         ax[1].plot(radar.index[index_index] ,radar['northing'][index_index],marker='.',linestyle='',label='radar', color=color[i])    
#         ax[2].plot(radar.index[index_index] ,radar['Z - Elevat'][index_index],marker='.',linestyle='',label='radar', color=color[i])
    
#     ax[2].set_xlim(date_range)
#     #ax[2].set_ylim(-525000,-510000)
#     ax[2].set_ylabel('(m.a.s.l.)')
#     #ax[2].legend()
#     #plt.savefig('radar_comparison_Line_%s'%file_name)

#%% KI --- plot by file and color by lines 
file_names = radar['File Name'].unique()
colors = ['aqua', 'beige', 'blue', 'darkgreen', 'ivory',
          'lime', 'magenta', 'orange', 'plum', 'red',
          'purple','silver','yellow','yellowgreen']

for file_name in file_names:
    index = radar['File Name'] == file_name

    fig,ax = plt.subplots(1, sharex=True, figsize=(40,20))
    fig.suptitle('File %s'%file_name)
    date_range = [pd.to_datetime('2022-08-11 00:00'),pd.to_datetime('2022-08-26 00:00')]
    
    ax.set_title('EASTING')
    ax.plot(gps_rover.index, gps_rover['easting'], marker='o',linestyle='',label='gps - rover', color='black') # - timedelta(hours=24)
    ax.plot(gps_hh.index, gps_hh['easting'], marker='.',linestyle='',label='gps - hand held', color='grey')
    
    line_name = radar[index]['Line Name'].unique()
    for line_name in line_name:
        sub_index = (radar['File Name'] == file_name)&(radar['Line Name'] == line_name)
        ax.plot(radar.index[sub_index] ,radar['easting'][sub_index],marker='.',linestyle='',label=line_name, color=np.random.choice(colors))
    
    ax.set_xlim(date_range)
    ax.set_ylim(-525000,-510000)
    ax.set_ylabel('(m)')
    ax.legend(fontsize=15)



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







    
    




















