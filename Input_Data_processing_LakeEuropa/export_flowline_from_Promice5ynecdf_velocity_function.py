

#!!! To enable gdal to run without issues, you need to create an environement through the conda prompt
# run this line: (gdal_env is the name you give it. this can be changed)
#   > conda create -n gdal_env python=3.6 gdal matplotlib netCDF4 seaborn pandas astropy spyder
#   > conda activate gdal_env
#   > sypder

#%%


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
from scipy.interpolate import interp1d




def get_netcdf_data(netcdf_file_path, data_type='Measures'): #

    #load netcdf data
    ds = nc.Dataset(netcdf_file_path)
    #display all the variables stored in the netcdf file
    print(ds.variables.keys())       
    data = {}
    
    if data_type == 'Promice':  
        
        #Velocity netcdf file from 
        #Solgaard, A. et al. Greenland ice velocity maps from the PROMICE project. Earth System Science Data 13, 3491–3512 (2021).
        #Downloaded in the Dataverse on this page: https://dataverse.geus.dk/dataverse/Ice_velocity/
        #"Multi-year Ice Velocity Mosaics for the Greenland Ice Sheet from Sentinel-1 Edition 1"
        #velocities are in  m/day
    
        #% extract x and y velocites arrays from netcdf   
        data['velocity_easting'] = ds['land_ice_surface_easting_velocity'][:][0]
        data['velocity_northing'] = ds['land_ice_surface_northing_velocity'][:][0]
        data['velocity_magnitude'] = ds['land_ice_surface_velocity_magnitude'][:][0]
        data['easting'] = ds['x'][:]
        data['northing'] = ds['y'][:]
        
    if data_type == 'Measures':
        #Velocity netcdf file from 
        #https://its-live.jpl.nasa.gov/
        
        # velocities are in meter per year
        # projection system: "WGS 84 / NSIDC Sea Ice Polar Stereographic North"
        
        #% extract x and y velocites arrays from netcdf   
        data['velocity_easting'] = ds['vx'][:]
        data['velocity_northing'] = ds['vy'][:]
        data['velocity_magnitude'] = ds['v'][:]
        data['easting'] = ds['x'][:]
        data['northing'] = ds['y'][:]        
    
    return data
   

def extract_flowline_position(data, 
                              time_correction_factor=1,
                              initial_position_easting=None, 
                              initial_position_northing=None, 
                              time_upstream=None, 
                              time_downstream=None,
                              extent=None, #[W,S,E,N]
                              # max_north=None, 
                              # max_south=None, 
                              # max_est=None, 
                              # max_west=None, 
                              ):

    # use rbs to create a special matrice to enable position searching
    rbs_easting = RectBivariateSpline(data['easting'], data['northing'][::-1], data['velocity_easting'][::-1].T) 
    rbs_northing = RectBivariateSpline(data['easting'], data['northing'][::-1], data['velocity_northing'][::-1].T) 
    rbs_magnitude = RectBivariateSpline(data['easting'], data['northing'][::-1], data['velocity_magnitude'][::-1].T)   
    
    #initiallize dictionnary
    keyList = ['coordinates_easting','coordinates_northing','velocity_easting','velocity_northing','velocity_magnitude']
    flowline = {key: [] for key in keyList}
    
    #extract initial position of the flow line
    position_easting = initial_position_easting
    position_northing = initial_position_northing 
    
    #loop through the flowline position finding
    #DOWNSTREAM
    for time in range(time_downstream): #3450
        #record how many years it runs
        #time_downstream = time+1
        #pull out the vx,vy components in the velocity fields in m/y for a specific point
        local_velocity_easting = rbs_easting.ev(position_easting, position_northing)*time_correction_factor
        #print(local_velocity_easting)
        local_velocity_northing = rbs_northing.ev(position_easting, position_northing)*time_correction_factor
        local_velocity_magnitude = rbs_magnitude.ev(position_easting, position_northing)*time_correction_factor#*30.4375
        #calculate position downstream for next timestep based on the velocity of the ice
        position_easting = position_easting + local_velocity_easting
        #print(position_easting)
        position_northing = position_northing + local_velocity_northing
        
        if extent==None:        
            flowline['coordinates_easting'].append(position_easting) 
            flowline['coordinates_northing'].append(position_northing)
            flowline['velocity_easting'].append(local_velocity_easting)
            flowline['velocity_northing'].append(local_velocity_northing)
            flowline['velocity_magnitude'].append(local_velocity_magnitude)
        else:
            south = extent[1]
            west = extent[0]
            north = extent[3]
            est = extent[2]
            if (position_easting >= west) and (position_easting <= est) and (position_northing <= north) and (position_northing >= south):                         
                flowline['coordinates_easting'].append(position_easting) 
                flowline['coordinates_northing'].append(position_northing)
                flowline['velocity_easting'].append(local_velocity_easting)
                flowline['velocity_northing'].append(local_velocity_northing)
                flowline['velocity_magnitude'].append(local_velocity_magnitude)
            else:
                print('out of extent boundary')
                break
          
            
# extent=[-530000,-1201200,-510896.671,-1190480.853])
     
    #extract initial position of the flow line
    position_easting = initial_position_easting
    position_northing = initial_position_northing 
    
    #UPSTREAM 
    for time in range(time_upstream): #3500
        #record how many years it runs
        #time_upstream = time+1
        #pull out the vx,vy components in the velocity fields in m/y for a specific point
        local_velocity_easting = rbs_easting.ev(position_easting, position_northing)*time_correction_factor #*30.4375
        local_velocity_northing = rbs_northing.ev(position_easting, position_northing)*time_correction_factor#*30.4375
        local_velocity_magnitude = rbs_magnitude.ev(position_easting, position_northing)*time_correction_factor#*30.4375
        #calculate position downstream for next timestep based on the velocity of the ice
        position_easting = position_easting-local_velocity_easting
        position_northing = position_northing-local_velocity_northing
        
        #calculate position upstream for next timestep based on the velocity of the ice
        #append upstream flowline position and velocity at the beginning of the list
        if extent==None:    
            flowline['coordinates_easting'].insert(0,position_easting) 
            flowline['coordinates_northing'].insert(0,position_northing)
            flowline['velocity_easting'].insert(0,local_velocity_easting)
            flowline['velocity_northing'].insert(0,local_velocity_northing)    
            flowline['velocity_magnitude'].insert(0,local_velocity_magnitude)        
        else:
            south = extent[1]
            west = extent[0]
            north = extent[3]
            est = extent[2]
            if (position_easting >= west) and (position_easting <= est) and (position_northing <= north) and (position_northing >= south):    
                flowline['coordinates_easting'].insert(0,position_easting) 
                flowline['coordinates_northing'].insert(0,position_northing)
                flowline['velocity_easting'].insert(0,local_velocity_easting)
                flowline['velocity_northing'].insert(0,local_velocity_northing)    
                flowline['velocity_magnitude'].insert(0,local_velocity_magnitude)    
            else:
                print('out of extent boundary')
                continue

    # inpterpolate to get a equidistant profile points   
    x = flowline['coordinates_easting']
    y = flowline['coordinates_northing']
    
    # Linear length on the line
    flowline['distance'] = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ))
    
           
    return flowline

def interpolate_flowline(flowline, 
                         spacing=10, 
                         csv_file_name='flowline.csv',
                         plot_figure=False,
                         max_length=None):
    # # inpterpolate to get a equidistant profile points   
    x = flowline['coordinates_easting']
    y = flowline['coordinates_northing']
    
    # # Linear length on the line
    # distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ))
    #print(distance)
    
    distance_total = flowline['distance'][-1]
    print('distance_total=', distance_total)
    distance = flowline['distance']/distance_total
    number_of_nodes = int(distance_total/spacing)
    print('number of nodes=', number_of_nodes)
    
    fx, fy = interp1d( distance, x ), interp1d( distance, y )  
    alpha = np.linspace(0, int(distance[-1]), number_of_nodes) #or use arange???
    #print('alpha',alpha)
    flowline_interp_easting, flowline_interp_northing = fx(alpha), fy(alpha)
    distance_between_nodes = distance_total/number_of_nodes
    print('distance between nodes =', distance_between_nodes, ' m')
    distance_array = np.linspace(0,distance_total,number_of_nodes)
    
    if plot_figure == True:
        plt.figure()
        plt.plot(x, y, '-')
        plt.plot(flowline_interp_easting, flowline_interp_northing, 'or')
        plt.axis('equal')
    
    df = pd.DataFrame({'easting':flowline_interp_easting,
                      'northing':flowline_interp_northing,
                      'distance':distance_array
                      })
    
    if max_length is not None:
        df = df.drop(df[df.distance>max_length].index)
        
    df.to_csv(csv_file_name) #'europa_flowline_coordinates_110m.csv'
    return df
    

def plot_map_flowline(flowline,
                      data,
                      extent=[-650250, 849750, -3349750, -649750],
                      xlim=[-5.6e5,-4.9e5],
                      ylim=[-1.26e6,-1.18e6],
                      vmin=[0,0.05]):

    years = range(len(flowline['velocity_easting']))
    fig, ax = plt.subplots()
    #plot velocity map 
    ax.imshow(data['velocity_magnitude'],extent=extent,vmin=vmin[0],vmax=vmin[1])
    #plot flow line
    sc = ax.scatter(flowline['coordinates_easting'],flowline['coordinates_northing'],c=years,cmap=cm.jet)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('years')
    #ax.plot(flowline_easting,flowline_northing, color='black')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# #%%

# fig,ax = plt.subplots(3)
# ax[0].plot(profile/1000, velocity_profile_magnitude, label='magnitude')
# ax[0].plot(profile/1000, velocity_profile_amplitude, label='calculated amplitude')
# ax[0].legend()

# ax[1].plot(profile/1000, velocity_profile_easting, label='velocity easting')
# ax[1].legend()

# ax[2].plot(profile/1000, velocity_profile_northing, label='velocity northing')
# ax[2].set_xlabel('Distance from ice divide (kn)')
# ax[2].legend()


    
#%% loop through the flowlines

#%% import data


#path = 'D:/Dropbox/RESEARCH/Qgreenland_LakeEuropa/QGreenland_v2.0.0/Glaciology/Ice sheet velocity/PROMICE Multi-year ice velocity/'
#filename = 'Promice_AVG5year.nc'
netcdf_promice_file_path = 'J:/QGreenland_v2.0.0/Additional/PROMICE Multi-year ice velocity/Promice_AVG5year.nc'
#netcdf_measures_file_path = 'J:/QGreenland_v2.0.0/Additional/MEaSUREs 120m composite velocity/GRE_G0120_0000.nc'
path_to_save = 'G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/FlowBand/Flowlines/'

data_promice = get_netcdf_data(netcdf_promice_file_path, data_type='Promice')

# #initial coordinates of the flowlines
# #coordinate system and projection: WGS 84 / NSIDC Sea Ice Polar Stereographic North EPSG:3413
# camp = [-516335.771, -1191434.196] #x and y
# start_flowline_2 = [-515762.315,-1192663.735]
# ice_divide = [-510900.089,-1190477.845]

# #extract initial position of the flow line
# position_easting = camp[0]
# position_northing = camp[1]

starting_points = pd.read_csv('G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/FlowBand/Flowlines/flowline_starting_points.csv')
starting_points['time_upstream'] = [3450, 3450, 3450, 3450, 3450, 3450, 3000 ]
starting_points['time_downstream'] = [3430, 3000, 3000, 3000, 3000, 2500, 800 ]

for i in [6]:#np.arange(len(starting_points)):
    
    
    flowline= extract_flowline_position(data_promice, 
                                  time_correction_factor=365,
                                  initial_position_easting=starting_points.easting[i], 
                                  initial_position_northing=starting_points.northing[i], 
                                  time_upstream=starting_points.time_upstream[i], 
                                  time_downstream=starting_points.time_downstream[i])
                                  #extent=[-530000,-1201200,-510896.671,-1190480.853])#3500
    
    plot_map_flowline(flowline, data_promice,
                      extent=[-650250, 849750, -3349750, -649750],
                      xlim=[-5.6e5,-4.9e5],
                      ylim=[-1.26e6,-1.18e6],
                      vmin=[0,0.05] )
    
    
    flowline_interp = interpolate_flowline(flowline, 
                             spacing=10,
                             max_length=25000,#☺m
                             csv_file_name=path_to_save+'promice_500m_europa_flowline_%s.csv'%starting_points.name[i])




#%%











# #data_measures = get_netcdf_data(netcdf_measures_file_path, data_type='Measures')

# flowline_camp = extract_flowline_position(data_promice, 
#                               time_correction_factor=365,
#                               initial_position_easting=camp[0], 
#                               initial_position_northing=camp[1], 
#                               time_upstream=3450, 
#                               time_downstream=3430)
#                               #extent=[-530000,-1201200,-510896.671,-1190480.853])#3500

# plot_map_flowline(flowline_camp, data_promice,
#                   extent=[-650250, 849750, -3349750, -649750],
#                   xlim=[-5.6e5,-4.9e5],
#                   ylim=[-1.26e6,-1.18e6],
#                   vmin=[0,0.05] )


# flowline_interp = interpolate_flowline(flowline_camp, 
#                          spacing=10,
#                          max_length=25000,#☺m
#                          csv_file_name=path_to_save+'promice_500m_europa_flowline_camp.csv')


   
# flowline_2 = extract_flowline_position(data_promice, 
#                               time_correction_factor=365,                                       
#                               initial_position_easting=start_flowline_2[0], 
#                               initial_position_northing=start_flowline_2[1], 
#                               time_upstream=3450, 
#                               time_downstream=2500)

# plot_map_flowline(flowline_2, data_promice,
#                   extent=[-650250, 849750, -3349750, -649750],
#                   xlim=[-5.6e5,-4.9e5],
#                   ylim=[-1.26e6,-1.18e6],
#                   vmin=[0,0.05] )

# interpolate_flowline(flowline_2, 
#                           spacing=10, 
#                           max_length=25000,#☺m
#                           csv_file_name=path_to_save+'promice_500m_europa_flowline_2.csv')


# flowline_ice_divide = extract_flowline_position(data_promice, 
#                               time_correction_factor=365,                                                
#                               initial_position_easting=ice_divide[0], 
#                               initial_position_northing=ice_divide[1], 
#                               time_upstream=3450, 
#                               time_downstream=6000)
  
# plot_map_flowline(flowline_ice_divide, data_promice,
#                   extent=[-650250, 849750, -3349750, -649750],
#                   xlim=[-5.6e5,-4.9e5],
#                   ylim=[-1.26e6,-1.18e6],
#                   vmin=[0,0.05] )


# interpolate_flowline(flowline_ice_divide, 
#                           max_length=25000,#☺m                
#                           spacing=10, 
#                           csv_file_name=path_to_save+'promice_500m_europa_flowline_ice_divide.csv')  

#%%


# #%%
# flowline_camp = extract_flowline_position(data_measures, 
#                               time_correction_factor=1,
#                               initial_position_easting=camp[0], 
#                               initial_position_northing=camp[1], 
#                               time_upstream=3450, 
#                               time_downstream=3500)
# plot_map_flowline(flowline_camp, data_measures,
#                   extent=[-650250, 849750, -3349750, -649750],
#                   xlim=[-5.6e5,-4.9e5],
#                   ylim=[-1.26e6,-1.18e6],
#                   vmin=[0,20] )
# interpolate_flowline(flowline_camp, 
#                          spacing=100, 
#                          csv_file_name=path_to_save+'measures_120m_europa_flowline_camp.csv')
# #%%
   
# flowline_2 = extract_flowline_position(data_measures, 
#                               time_correction_factor=1,                                       
#                               initial_position_easting=flowline_2[0], 
#                               initial_position_northing=flowline_2[1], 
#                               time_upstream=3450, 
#                               time_downstream=3500)
# plot_map_flowline(flowline_2, data_measures,
#                   extent=[-650250, 849750, -3349750, -649750],
#                   xlim=[-5.6e5,-4.9e5],
#                   ylim=[-1.26e6,-1.18e6],
#                   vmin=[0,20] )
# interpolate_flowline(flowline_2, 
#                          spacing=100, 
#                          csv_file_name=path_to_save+'measures_120m_europa_flowline_2.csv')

# #%%  
# flowline_ice_divide = extract_flowline_position(data_measures, 
#                               time_correction_factor=1,                                                
#                               initial_position_easting=ice_divide[0], 
#                               initial_position_northing=ice_divide[1], 
#                               time_upstream=3450, 
#                               time_downstream=5000)
  
# plot_map_flowline(flowline_ice_divide, data_measures,
#                   extent=[-650250, 849750, -3349750, -649750],
#                   xlim=[-5.6e5,-4.9e5],
#                   ylim=[-1.26e6,-1.18e6],
#                   vmin=[0,20] )


# interpolate_flowline(flowline_ice_divide, 
#                          spacing=100, 
#                          csv_file_name=path_to_save+'measures_120m_europa_flowline_ice_divide.csv')  


















































