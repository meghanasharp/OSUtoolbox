# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 12:31:56 2022

@author: trunzc
"""


#!!! To enable gdal to run without issues, you need to create an environement through the conda prompt
# run this line: (gdal_env is the name you give it. this can be changed)
#   > conda create -n gdal_env python=3.6 gdal matplotlib netCDF4 seaborn pandas astropy spyder
#   > conda activate gdal_env
#   > sypder



#extract data on flow line path
from astropy.io import fits
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