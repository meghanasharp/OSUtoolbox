# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 16:17:38 2022

@author: trunzc
"""

#extract data on flow line path
from astropy.io import fits
from scipy.io import savemat
from astropy.utils.data import get_pkg_data_filename
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from osgeo import gdal,ogr
from matplotlib import pyplot as plt
import matplotlib.cm as cm
#import matplotlib.cm as cm
import numpy as np
import netCDF4 as nc
import seaborn as sns
import pandas as pd
#from math import radians, cos, sin, asin, sqrt
#from scipy.spatial import KDTree
from scipy.interpolate import RectBivariateSpline    # interpolate grid data to point coordinates
#from scipy.interpolate import interp1d


def get_xy_array_geotiff(rds):
    """
    Gets the easting and northing of the geotiff. 
    Code comes from Christian Wild (Oregon State University)

    Parameters
    ----------
    rds : gdal.Dataset
        Geotiff loaded with gdal.Open('geotiff.tif').

    Returns
    -------
    easting : Array of float64
        x coordinates of the grid (pixel centered).
    northing : Array of float64
        y coordinates of the grid (pixel centered).

    """

    # get some more info about the grid
    nx = rds.RasterXSize
    ny = rds.RasterYSize

    geotransform = rds.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    
    endX = originX + pixelWidth * nx
    endY = originY + pixelHeight * ny
    
    easting = np.arange(nx) * pixelWidth + originX + pixelWidth/2.0 # pixel center
    northing = np.arange(ny) * pixelHeight + originY + pixelHeight/2.0 # pixel center
    
    #grid_lons, grid_lats = np.meshgrid(lons, lats)
    return easting,northing

def get_geotiff_data(path_to_geotiff,interpolation_iteration=4):  
    rds = gdal.Open(path_to_geotiff)
    band = rds.GetRasterBand(1)
    data = {}  
    data['grid'] = band.ReadAsArray()
    data['grid'][data['grid']==-9999]=np.nan
    data['easting'],data['northing']=get_xy_array_geotiff(rds)
    
    #interpolate nans
    # We smooth with a Gaussian kernel with x_stddev=1 (and y_stddev=1)
    # It is a 9x9 array
    kernel = Gaussian2DKernel(x_stddev=1)
    # create a "fixed" image with NaNs replaced by interpolated values
    for i in np.arange(interpolation_iteration):
        
        data['grid_interpolated'] = interpolate_replace_nans(data['grid'], kernel)
    data['grid_interpolated'] [np.isnan(data['grid_interpolated'] )] = -9999
    #data['grid_interpolated'] = np.nan_to_num(data['grid_interpolated'], nan=-9999)

    return data

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
        
    
    #% extract arrays from netcdf       
    if data_type == 'Bedmachine':  
        data['surface'] = ds['surface'][:]
        data['thickness'] = ds['thickness'][:]
        data['bed'] = ds['bed'][:]
        data['easting'] = ds['x'][:]
        data['northing'] = ds['y'][:]   
        
    #% extract arrays from netcdf       
    if data_type == 'HeatFlux':  
        data['correction'] = ds['correction'][:]
        data['correction_uncertainty'] = ds['correction_uncertainty'][:]
        data['easting'] = ds['x'][:]
        data['northing'] = ds['y'][:]   
          
    return data

# def get_data_from_profile(flowline, data, layer_name, plot=False):
#     rbs_surface = RectBivariateSpline(data['easting'], data['northing'][::-1], data[layer_name][::-1].T)
    
#     profile = rbs_surface.ev(flowline.easting, flowline.northing)
#     if plot==True:
#         fig,ax = plt.subplots()
#         ax.plot(flowline.distance,profile_ice_surface)
#     return profile




def get_data_from_profile(flowline, data, layer_name):
    
    rbs_surface = RectBivariateSpline(data['easting'], data['northing'][::-1], data[layer_name][::-1].T) 
    profile = rbs_surface.ev(flowline.easting, flowline.northing)
    return profile


def plot_map(fig, ax, 
             grid=[],
             easting=[],
             northing=[],
             xlim=[-560000,-490000],
             ylim=[-1260000,-1180000],
             vlim=None):
    
   # fig, ax = plt.subplots()
    extent=[easting[0], easting[-1],northing[-1], northing[0]]
    #years = range(len(flowline['velocity_easting']))
    
    #plot velocity map 
    if vlim==None:
        sc = ax.imshow(grid,extent=extent)
    else:
        sc = ax.imshow(grid,extent=extent,vmin=vlim[0],vmax=vlim[1])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    bar = plt.colorbar(sc)
    
    
def plot_flowline(fig, ax,profile):
    years = range(len(profile))
    sc = ax.scatter(profile['easting'],profile['northing'],c=years,cmap=cm.jet)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('years')
    


def find_extent_on_profile(extent_lower_left_corner, extent_upper_right_corner):
    area = profile.distance[(profile.easting > extent_lower_left_corner[0]) & 
                                 (profile.easting < extent_upper_right_corner[0]) & 
                                 (profile.northing > extent_lower_left_corner[1]) & 
                                 (profile.northing < extent_upper_right_corner[1]) ]
    
    return area.reset_index(drop=True)

#%%



# Load flowline
# profile = pd.read_csv('G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/FlowBand/Flowlines/promice_500m_europa_flowline_camp.csv', 
#                        index_col=0)

# profile = pd.read_csv('G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/FlowBand/Flowlines/promice_500m_europa_flowline_2.csv', 
#                        index_col=0)

profile = pd.read_csv('G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/FlowBand/Flowlines/promice_500m_europa_flowline_camp.csv', 
                        index_col=0)

lake_extent_sw = [-521139.211,-1194232.358]
lake_extent_ne = [-514887.923,-1190764.258]

radar_extent_sw = [-522520.647,-1194383.272]
radar_extent_ne = [-512374.638,-1190389.877]    
lake_area = find_extent_on_profile(lake_extent_sw, lake_extent_ne)    

radar_area = find_extent_on_profile(radar_extent_sw, radar_extent_ne)   
    
#!! make this a function

#Load data from tiffs and netcdf

#LOAD WEATHER DATA
####################

racmo_temperature = get_geotiff_data('J:/QGreenland_v2.0.0/Regional climate models/RACMO model output/Annual mean temperature at 2m 1958-2019 (1km)/racmo_t2m.tif')
racmo_precipitation = get_geotiff_data('J:/QGreenland_v2.0.0/Regional climate models/RACMO model output/Total precipitation 1958-2019 (1km)/racmo_precip.tif')
racmo_runoff = get_geotiff_data('J:/QGreenland_v2.0.0/Regional climate models/RACMO model output/Runoff 1958-2019 (1km)/racmo_runoff.tif')
racmo_melt = get_geotiff_data('J:/QGreenland_v2.0.0/Regional climate models/RACMO model output/Snowmelt 1958-2019 (1km)/racmo_snowmelt.tif')
racmo_sublimation = get_geotiff_data('J:/QGreenland_v2.0.0/Regional climate models/RACMO model output/Sublimation 1958-2019 (1km)/racmo_subl.tif')
racmo_snowdrift = get_geotiff_data('J:/QGreenland_v2.0.0/Regional climate models/RACMO model output/Snow drift erosion 1958-2019 (1km)/racmo_sndiv.tif')


#LOAD HEAT MAP
###################
heatflux = get_geotiff_data('J:/QGreenland_v2.0.0/Geophysics/Heat flux/Heat flow (Colgan et al.)/Flow from multiple observations (55km)/geothermal_heat_flow_map.tif')
# topocorrection = get_netcdf_data('J:/QGreenland_v2.0.0/Additional/GeothermalHeatFlux_colgan2021JGR/TopoHeat_Greenland_20210224.nc', 
#                              data_type='HeatFlux')

#LOAD TOPO DATA
#################

bedmachine = get_netcdf_data('J:/QGreenland_v2.0.0/Additional/BedMachineGreenland_V5/BedMachineGreenland-v5.nc', 
                             data_type='Bedmachine')
#arcticdem_2m = get_geotiff_data('J:/QGreenland_v2.0.0/Additional/Arctic DEM/Arctic_DEM_mosaic_merged_2m.tif') #this is too big
arcticdem_10m = get_geotiff_data('J:/QGreenland_v2.0.0/Additional/Arctic DEM/Arctic_DEM_mosaic_merged_10m.tif')
arcticdem_32m = get_geotiff_data('J:/QGreenland_v2.0.0/Additional/Arctic DEM/Arctic_DEM_mosaic_merged_32m.tif')


#LOAD VELOCITIES
###################

promice_velocity = get_netcdf_data('J:/QGreenland_v2.0.0/Additional/PROMICE Multi-year ice velocity/Promice_AVG5year.nc',
                               data_type='Promice')
measures_velocity = get_netcdf_data('J:/QGreenland_v2.0.0/Additional/MEaSUREs 120m composite velocity/GRE_G0120_0000.nc', 
                                data_type='Measures')



#%%extract profiles


profile['racmo_2m_air_temperature_kelvin'] = get_data_from_profile(profile, racmo_temperature, 'grid_interpolated')
profile['racmo_total_precipitation_mm_water_equivalent']  = get_data_from_profile(profile, racmo_precipitation, 'grid_interpolated')
profile['racmo_runoff_mm_water_equivalent']  = get_data_from_profile(profile, racmo_runoff, 'grid_interpolated')
profile['racmo_snowmelt_mm_water_equivalent']  = get_data_from_profile(profile, racmo_melt, 'grid_interpolated')
profile['racmo_sublimation_mm_water_equivalent']  = get_data_from_profile(profile, racmo_sublimation, 'grid_interpolated')
profile['racmo_snowdrift_mm_water_equivalent']  = get_data_from_profile(profile, racmo_snowdrift, 'grid_interpolated')

profile['heat_flux']  = get_data_from_profile(profile, heatflux, 'grid_interpolated') #mW/m2

#profile['total_melt']
profile['bedmachine_ice_surface_elevation_masl']  = get_data_from_profile(profile, bedmachine, 'surface')
profile['bedmachine_ice_thickness_m']  = get_data_from_profile(profile, bedmachine, 'thickness')
profile['bedmachine_bed_elevation_masl']  = get_data_from_profile(profile, bedmachine, 'bed')

#profile['arcticdem_2m_ice_surface_elevation_masl']  = get_data_from_profile(profile, arcticdem_2m, 'grid_interpolated')
profile['arcticdem_10m_ice_surface_elevation_masl']  = get_data_from_profile(profile, arcticdem_10m, 'grid_interpolated')
profile['arcticdem_32m_ice_surface_elevation_masl']  = get_data_from_profile(profile, arcticdem_32m, 'grid_interpolated')

profile['promice_velocity_mperday']  = get_data_from_profile(profile, promice_velocity, 'velocity_magnitude')
profile['measures_velocity_mperyear']  = get_data_from_profile(profile, measures_velocity, 'velocity_magnitude')

#this calculation is based of Kiya and Georgia. its missing drift erosion
profile['racmo_total_mass_loss'] = profile['racmo_runoff_mm_water_equivalent']-profile['racmo_sublimation_mm_water_equivalent']
# for christian's model
profile['racmo_mass_balance'] = profile['racmo_total_precipitation_mm_water_equivalent'] - profile['racmo_total_mass_loss']

profile.to_csv('profile_data_lake_europa.csv')

#%% SAVE TO .mat

#tranform pandas dataframe to dictionnary
#profile.to_dict('list')

#mmWE/yr to mIE/yr (based on Kiya and Georgia's code Edit_inputfiles_Hiawatha.m')
water_to_ice_eq = 0.001/0.917
mwatt_to_watt = 0.001
mperday_to_mperyear = 365

# List of Keys
keylist = ['Accum_yr_input', 'AnnualMelt_yr_input', 'BedElev_input', 'Gflux_input', 'Icethick_input', 'SurfTemp_input', 'velocity', 'Width_input', 'X_input']

matdic = {key: None for key in keylist}
matdic['Accum_yr_input'] = profile['racmo_total_precipitation_mm_water_equivalent'].values
# why does the sublimation have negative values
matdic['AnnualMelt_yr_input'] = (profile['racmo_runoff_mm_water_equivalent'].values * water_to_ice_eq + 
                                 profile['racmo_sublimation_mm_water_equivalent'].values * water_to_ice_eq + 
                                 profile['racmo_snowdrift_mm_water_equivalent'].values * water_to_ice_eq )
#in m.a.s.l.
matdic['BedElev_input'] = profile['bedmachine_bed_elevation_masl'].values
#in W/m^2
#matdic['Gflux_input'] = profile['heat_flux'].values  *mwatt_to_watt
matdic['Gflux_input'] = np.zeros(len(profile['distance'])) + 0.1 #to make values similar to hiwatha
#in m
matdic['Icethick_input'] = profile['bedmachine_ice_thickness_m'].values

matdic['SurfTemp_input'] = profile['racmo_2m_air_temperature_kelvin'].values
matdic['velocity'] = profile['promice_velocity_mperday'].values*365
#constant width of a thousand meter
matdic['Width_input'] = np.ones(len(profile['distance']))*1000 
matdic['X_input'] = profile['distance'].values

# for key in keylist:
#     matdic[key] = np.vstack(matdic[key])

# # GenerateSlidingInputs
# # Mike Wolovick, 1/12/2022
# # This is a quick script for generating the sliding law inputs for the
# # flowline model -slightly edited by Georgia to be in the wrapper file

# # PARAMETERS:
# # Assumed slip fraction:
# slipfrac=.1;              % unitless

# # Wavelengths:
#     wavelength_geom=1e3;     % m
#     wavelength_slip=5e3;     % m

# # Minimum stress scale:
#     minstressscale=1e3;      % Pa

# # Other parameters:
#     rho_i=917;               % kg/m^3
#     g=9.8;                   % m/s^2


savemat('initial_inputs_europa.mat',matdic, oned_as='column')

#%% Plot data for flowband model inputs for AGU poster

fig,(ax1,ax2,ax3,ax4) = plt.subplots(4, sharex=True, figsize=(8,6))#, gridspec_kw={'height_ratios': [2, 1]})

ax1.plot(matdic['X_input'], matdic['SurfTemp_input'] , label='Surface temperature')

ax2.plot(matdic['X_input'], matdic['AnnualMelt_yr_input'], label='Precipitation - RACMO')
ax2.plot(matdic['X_input'], profile.racmo_runoff_mm_water_equivalent, label='Runoff - RACMO')

ax3.fill_between(matdic['X_input'], profile.bedmachine_ice_surface_elevation_masl, profile.bedmachine_bed_elevation_masl, alpha=0.3)
ax3.plot(matdic['X_input'], matdic['BedElev_input']+matdic['Icethick_input'] , label='Ice elevation - Bedmachine v5')
ax3.plot(matdic['X_input'], matdic['BedElev_input'] , label='Bed elevation - Bedmachine v5')

ax4.plot(matdic['X_input'], matdic['velocity'], label='Mean surface velocity - Promice')

ax1.axvspan(lake_area[0],lake_area[len(lake_area)-1], alpha=0.3, color='cyan', label='minimum lake extent')
ax1.axvspan(radar_area[0],radar_area[len(radar_area)-1], alpha=0.3, color='grey', label='radar extent')

ax2.axvspan(lake_area[0],lake_area[len(lake_area)-1], alpha=0.3, color='cyan')
ax2.axvspan(radar_area[0],radar_area[len(radar_area)-1], alpha=0.3, color='grey')

ax3.axvspan(lake_area[0],lake_area[len(lake_area)-1], alpha=0.3, color='cyan')
ax3.axvspan(radar_area[0],radar_area[len(radar_area)-1], alpha=0.3, color='grey')

ax4.axvspan(lake_area[0],lake_area[len(lake_area)-1], alpha=0.3, color='cyan')
ax4.axvspan(radar_area[0],radar_area[len(radar_area)-1], alpha=0.3, color='grey')


ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()



ax1.set_ylabel('(k)')
ax2.set_ylabel('(mm w.e.)')
ax3.set_ylabel('(m.a.s.l.)')
ax4.set_ylabel('(m/year)')


#%%

fig,(ax1,ax2,ax3,ax4, ax5) = plt.subplots(5, sharex=True)

ax1.plot(profile.distance,profile.racmo_2m_air_temperature_kelvin, label='Air temp')

ax2.plot(profile.distance,profile.racmo_total_precipitation_mm_water_equivalent, label='Precipitation - RACMO')
ax2.plot(profile.distance,profile.racmo_runoff_mm_water_equivalent, label='Runoff - RACMO')
ax2.plot(profile.distance,profile.racmo_snowmelt_mm_water_equivalent, label='Snow melt - RACMO')
ax2.plot(profile.distance,profile.racmo_sublimation_mm_water_equivalent, label='Sublimation - RACMO')

ax3.plot(profile.distance,profile.bedmachine_ice_thickness_m, label='Ice thickness - Bedmachine v5')

ax4.fill_between(profile.distance, profile.bedmachine_ice_surface_elevation_masl, profile.bedmachine_bed_elevation_masl, alpha=0.3)
ax4.plot(profile.distance,profile.bedmachine_ice_surface_elevation_masl, label='Ice elevation - Bedmachine v5')
ax4.plot(profile.distance,profile.bedmachine_bed_elevation_masl, label='Bed elevation - Bedmachine v5')

ax5.plot(profile.distance,profile.promice_velocity_mperday*365, label='Mean surface velocity - Promice')
ax5.plot(profile.distance,profile.measures_velocity_mperyear, label='Mean surface velocity - Measures')

ax1.axvspan(lake_area[0],lake_area[len(lake_area)-1], alpha=0.3, color='cyan', label='minimum lake extent')
ax1.axvspan(radar_area[0],radar_area[len(radar_area)-1], alpha=0.3, color='grey', label='radar extent')

ax2.axvspan(lake_area[0],lake_area[len(lake_area)-1], alpha=0.3, color='cyan')
ax2.axvspan(radar_area[0],radar_area[len(radar_area)-1], alpha=0.3, color='grey')

ax3.axvspan(lake_area[0],lake_area[len(lake_area)-1], alpha=0.3, color='cyan')
ax3.axvspan(radar_area[0],radar_area[len(radar_area)-1], alpha=0.3, color='grey')

ax4.axvspan(lake_area[0],lake_area[len(lake_area)-1], alpha=0.3, color='cyan')
ax4.axvspan(radar_area[0],radar_area[len(radar_area)-1], alpha=0.3, color='grey')

ax5.axvspan(lake_area[0],lake_area[len(lake_area)-1], alpha=0.3, color='cyan')
ax5.axvspan(radar_area[0],radar_area[len(radar_area)-1], alpha=0.3, color='grey')

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax5.legend()


ax1.set_ylabel('(k)')
ax2.set_ylabel('(mm w.e.)')
ax3.set_ylabel('(m)')
ax4.set_ylabel('(m.a.s.l.)')
ax5.set_ylabel('(m/year)')

#%%

#select extent
# xlim=[-560000,-490000]
# ylim=[-1260000,-1180000]
# index_easting = np.where((racmo_melt['easting'] > xlim[0]) & (racmo_melt['easting'] < xlim[1]))
# index_northing = np.where((racmo_melt['northing'] > ylim[0]) & (racmo_melt['northing'] < ylim[1]))


camp = [-516335.771, -1191434.196]


xlim = [-560000,-490000]
ylim = [-1260000,-1180000]

fig,ax = plt.subplots()
plot_map(fig, ax,
         grid = racmo_temperature['grid_interpolated'],
         easting = racmo_temperature['easting'],
         northing = racmo_temperature['northing'],
         xlim = xlim,
         ylim = ylim,
         vlim=[250,260])
plot_flowline(fig,ax, profile)
plt.title('temperature')

ax.plot(camp[0],camp[1],marker='x',markersize=12,color='black')

fig,ax = plt.subplots()
plot_map(fig, ax,
         grid = racmo_precipitation['grid_interpolated'],
         easting = racmo_precipitation['easting'],
         northing = racmo_precipitation['northing'],
         xlim = xlim,
         ylim = ylim,
         vlim=[0,750])
plot_flowline(fig,ax, profile)
plt.title('precipitation')

ax.plot(camp[0],camp[1],marker='x',markersize=12,color='black')

fig,ax = plt.subplots()
plot_map(fig, ax,
         grid = racmo_runoff['grid_interpolated'],
         easting = racmo_runoff['easting'],
         northing = racmo_runoff['northing'],
         xlim = xlim,
         ylim = ylim,
         vlim=[0,750])
plot_flowline(fig,ax, profile)
plt.title('runoff')

ax.plot(camp[0],camp[1],marker='x',markersize=12,color='black')

fig,ax = plt.subplots()
plot_map(fig, ax,
         grid = racmo_melt['grid_interpolated'],
         easting = racmo_melt['easting'],
         northing = racmo_melt['northing'],
         xlim = xlim,
         ylim = ylim,
         vlim=[0,750])
plot_flowline(fig,ax, profile)
plt.title('melt')

fig,ax = plt.subplots()
plot_map(fig, ax,
         grid = racmo_sublimation['grid_interpolated'],
         easting = racmo_sublimation['easting'],
         northing = racmo_sublimation['northing'],
         xlim = xlim,
         ylim = ylim,
         vlim=[0,750])
plot_flowline(fig,ax, profile)
plt.title('sublimation')

ax.plot(camp[0],camp[1],marker='x',markersize=12,color='black')

#%%

# rbs_surface = RectBivariateSpline(racmo_temperature['easting'], racmo_temperature['northing'][::-1], racmo_temperature['grid_interpolated'][::-1].T)

# test = rbs_surface.ev(profile.easting, profile.northing)

# fig,ax = plt.subplots()
# ax.plot(profile.distance,test)

    
    
    
    
    
    
    
