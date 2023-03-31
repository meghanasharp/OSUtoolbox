# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 13:15:06 2023

@author: trunzc
"""

# Import libraries
import glob
import pandas as pd
import numpy as np
#import pylab as pl
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import seaborn as sns
import pyproj
from math import radians, cos, sin, asin, sqrt
from scipy import stats



path = 'C:/Users/truc3007/Dropbox/NEIGE/SWE-VENT/'
#path = 'C:/Users/celia/Dropbox/NEIGE/SWE-VENT/'

#def load_data():
    
def haversine(lon1, lat1, lon2, lat2):
    '''
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Parameters
    ----------
    lon1 : float
        Longitudinal coordinate of point 1 in decimal degrees.
    lat1 : float
        Latitudinal coordinate of point 1 in decimal degrees.
    lon2 : float
        Longitudinal coordinate of point 2 in decimal degrees.
    lat2 : float
        Latitudinal coordinate of point 2 in decimal degrees.
    Returns
    -------
    float
        Distance between the two points in kilometers.
    '''

    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    distance_km = 6371* c
    return distance_km



#Projection: 
#in degree = 4326
# EPSG:3413 - WGS 84 / NSIDC Sea Ice Polar Stereographic North
proj_to_degree = pyproj.Transformer.from_crs(3413, 4326, always_xy=True)
proj_to_stereo = pyproj.Transformer.from_crs(4326, 3413, always_xy=True)

# xlim = [-69.6,-68.05]
# ylim = [77.7,78.35]

xlim = [-525000,-510000]
ylim = [-1200000,-1180000]

#IMPORT DATA
##############################################################################
##############################################################################

#load map plotting features
#radar data lines from the 2022 summer field season exported from QGIS
radar = pd.read_csv('J:/csv_for_2d_plotting/radar_lines.csv', index_col=0)
#lake boundaries from the texas
lake = pd.read_csv('J:/csv_for_2d_plotting/naqlk_area.csv')

# add coordinates in degree
radar['longitude_start'], radar['latitude_start'] = proj_to_degree.transform(radar.easting_start, radar.northing_start)
radar['longitude_end'], radar['latitude_end'] = proj_to_degree.transform(radar.easting_end, radar.northing_end)
lake['longitude'], lake['latitude'] = proj_to_degree.transform(lake.easting, lake.northing)


    


#Load icesat data
##########################################################

# Get CSV files list from a folder

csv_files = glob.glob("J:/IceSat/FromAltimetry_large_area/*.csv")

# Read each CSV file into DataFrame
# This creates a list of dataframes
icesat_list = (pd.read_csv(file) for file in csv_files)
# Concatenate all DataFrames
icesat   = pd.concat(icesat_list, ignore_index=True)
# extract date from filename
icesat['date'] = [x[16:20]+'-'+x[20:22]+'-'+x[22:24] for x in icesat.file_name]
#icesat['date'] = [x[16:24] for x in icesat.file_name]
# add easting and northing
icesat['easting'], icesat['northing'] = proj_to_stereo.transform(icesat.longitude, icesat.latitude)

#remove data with low confidence
icesat.drop(icesat[icesat.atl06_quality_summary==1].index, inplace=True)
icesat.drop(icesat[(icesat.easting < xlim[0]) |
                    (icesat.easting > xlim[1]) |
                    (icesat.northing < ylim[0]) |
                    (icesat.northing > ylim[1])
                    ].index, inplace=True)

#reset index
icesat = icesat.reset_index(drop=True)


#extract all the unique values for icesat metadata
filenames = icesat.file_name.unique()
beams = icesat.beam.unique()
tracks = icesat.track_id.unique()


#divide easting and northing coordinates by 1000 for easier plots
radar.easting_start = radar.easting_start/1000
radar.easting_end = radar.easting_end/1000
radar.northing_start = radar.northing_start/1000
radar.northing_end = radar.northing_end/1000

#transform radar lines for plotting
radar_lines = []
for i in radar.index:
    radar_lines.append([(radar.easting_start[i],radar.northing_start[i]),
                        (radar.easting_end[i],radar.northing_end[i])])

lake.easting = lake.easting/1000
lake.northing = lake.northing/1000

icesat.easting = icesat.easting/1000
icesat.northing = icesat.northing/1000

xlim = [xlim[0]/1000,xlim[1]/1000]
ylim = [ylim[0]/1000,ylim[1]/1000]

#%% find similar path -- uncomment if necesseray to do the figures or table again

# #create x axis for fit function. Same for all.
# x_fit = np.linspace(xlim[0],[xlim[1]], 100)

# # for a certain position x, which y is similar?
# #track = tracks[0]

# #iterate through the tracks, but using only the same beam. 
# #since the beams have the same spacing between each other, 
# #figuring out which day correlates with which can be made with one beam
# for track in tracks:
#     beam = beams[-1]  
#     dates = icesat[(icesat.track_id==track) & (icesat.beam==beam)].date.unique()
#     table = pd.DataFrame(columns=dates, index=dates)
    
#     for date1 in dates:
#         #calculates the fitting function going through a single track and beam for the day to compare to:        
#         icesat_selection1 = icesat[(icesat.track_id==track) & (icesat.beam==beam) & (icesat.date==date1)]
#         x1 = icesat_selection1.easting
#         y1 = icesat_selection1.northing
#         model1 = np.poly1d(np.polyfit(x1, y1, 1))
#         y_fit1 = model1(x_fit).flatten()
        
#         for date2 in dates:
#             icesat_selection2 = icesat[(icesat.track_id==track) & (icesat.beam==beam) & (icesat.date==date2)]
#             x2 = icesat_selection2.easting
#             y2 = icesat_selection2.northing
#             model = np.poly1d(np.polyfit(x2, y2, 1))
#             y_fit2 = model(x_fit).flatten()
            
#             #calculates the vertical deviation from the two fitting function, in km
#             # less than a km is probably okay. more not so much
#             deviation = np.round(np.mean(y_fit1-y_fit2),2)#np.round(np.std(y_fit1,y_fit2),2)
            
#             #save deviation in table:
#             table[date1][date2] = deviation
#             table.to_csv('J:/IceSat/Figures/table_correlation_day_track%s.csv'%track)
             
#             #plot
#             fig,ax = plt.subplots(1)
#             ax.set_title('Track ID=%s / Beam=%s / mean deviation=%sm '%(track,beam,deviation))
#             #ax.fill_between(lake.easting,lake.northing)
#             ax.plot(x_fit,y_fit1, linestyle='--', color='grey', label='fitting functions')
#             ax.plot(x_fit,y_fit2, linestyle='--', color='grey')
#             ax.add_collection(mc.LineCollection(radar_lines, linewidths=2, color='black'))
#             ax.plot(icesat_selection1.easting, icesat_selection1.northing, label=date1)
#             ax.plot(icesat_selection2.easting, icesat_selection2.northing, label=date2)
#             ax.legend()
#             plt.savefig('J:/IceSat/Figures/path_correlations/corr_track=%s_Beam=%s_%svs%s.png'%(track,beam,date1,date2))
#             plt.close()

to_drop = pd.DataFrame(list(zip(['2018-12-31','2019-12-29','2019-01-16','2019-03-15','2018-12-02','2019-03-03','2018-10-30','2019-01-29','2019-02-14'],
[49,49,286,1170,994,994,491,491,728])), columns=['date','track'])

for i in to_drop.index:
    icesat.drop(icesat[(icesat.track_id==to_drop.track[i])&(icesat.date==to_drop.date[i])].index, inplace=True) 

#%% 
# PLOTS
############################################################################
############################################################################

for track in tracks:

    data = icesat[(icesat.track_id==track)]
    
    fig,ax = plt.subplots(4, gridspec_kw={'height_ratios':[4,1,1,1]}, figsize=(10,15), sharex=True)  
    fig.suptitle('Track %s'%track)
    
    #plot 2d visualization of tracks
    ax[0].add_collection(mc.LineCollection(radar_lines, linewidths=2))
    sns.scatterplot(data=data, x='easting', y='northing', hue='beam', linewidth=0, palette = 'icefire', alpha=0.5, ax=ax[0])
    
    #plot beam 1 r and l
    ax[1].set_title('Beams gt1r and gt1l')
    sns.scatterplot(data=data[data.beam==beams[4]], x='easting', y='h_li', hue='date',linewidth=0, palette = 'YlOrBr', alpha=0.5,  ax=ax[1])
    sns.scatterplot(data=data[data.beam==beams[5]], x='easting', y='h_li', hue='date',linewidth=0, palette = 'YlOrBr', alpha=0.5, linestyle='--',  ax=ax[1], legend=False)
    
    #plot beam 2 r and l
    ax[2].set_title('Beams gt2r and gt2l')
    sns.scatterplot(data=data[data.beam==beams[2]], x='easting', y='h_li', hue='date',linewidth=0, palette = 'Greys', alpha=0.5,  ax=ax[2], legend=False)
    sns.scatterplot(data=data[data.beam==beams[3]], x='easting', y='h_li', hue='date',linewidth=0, palette = 'Greys', alpha=0.5, linestyle='--',  ax=ax[2], legend=False)
    
    #plot beam 3 r and l
    ax[3].set_title('Beams gt3r and gt3l')
    sns.scatterplot(data=data[data.beam==beams[0]], x='easting', y='h_li', hue='date',linewidth=0, palette = 'GnBu', alpha=0.5,  ax=ax[3], legend=False)
    sns.scatterplot(data=data[data.beam==beams[1]], x='easting', y='h_li', hue='date',linewidth=0, palette = 'GnBu', alpha=0.5, linestyle='--',  ax=ax[3], legend=False)
    
    ax[0].legend(fontsize=8, frameon=False)
    ax[1].legend(fontsize=8, frameon=False)
    
    sns.despine(ax=ax[0], bottom=True)
    sns.despine(ax=ax[1], bottom=True)
    sns.despine(ax=ax[2], bottom=True)
    sns.despine(ax=ax[3], bottom=False)
    
    ax[0].tick_params(bottom=False)
    ax[1].tick_params(bottom=False)
    ax[2].tick_params(bottom=False)
    
    ax[1].set(ylabel='Elevation')
    ax[2].set(ylabel='Elevation')
    ax[3].set(ylabel='Elevation')
    
    #ax[0].autoscale()
    #ax[0].set_aspect('equal')#, adjustable='datalim')
    #ax[0].margins(0.1)
    
    plt.savefig('J:/IceSat/Figures/Icesat_LakeEuropa_compare_track%s'%track)









#%% Plot where there is data
#fig,ax = plt.subplots()
#sns.scatterplot(data=icesat, x='longitude', y='latitude', hue='date')

#%%

for beam in beams:
    fg = sns.FacetGrid(data=icesat[icesat.beam==beam], hue='date', aspect=1.61)
    fg.map(plt.scatter,'longitude', 'h_li').add_legend()
    
    
#%%

fg = sns.FacetGrid(data=icesat, 
                   hue='date', 
                   #col= 'file_name',
                   row='beam',
                   aspect=1.61)
fg.map(plt.scatter,'longitude', 'h_li').add_legend()

#%%
for beam in beams:
    
    fig,ax = plt.subplots(2)  
    sns.scatterplot(data=icesat[icesat.beam==beam], x='longitude', y='latitude', hue='date', linewidth=0, palette = 'icefire', alpha=0.5, ax=ax[0])
    sns.scatterplot(data=icesat[icesat.beam==beam], x='longitude', y='h_li', hue='date',linewidth=0, palette = 'icefire', alpha=0.5,  ax=ax[1])

    ax[0].plot([radar.x_start,radar.x_end],[radar.y_start, radar.y_end])
    # for i in np.arange(len(radar)):
    #     ax[0].plot()




#%%

import re, seaborn as sns
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

x = icesat.longitude
y = icesat.latitude
z = icesat.h_li

# axes instance
fig = plt.figure(figsize=(6,6))
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

# get colormap from seaborn
cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())

# plot
sc = ax.scatter(x, y, z, s=40, c=x, marker='o', cmap=cmap, alpha=1)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# legend
plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

# save
#plt.savefig("scatter_hue", bbox_inches='tight')