# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 13:15:06 2023

@author: trunzc
"""

# Import libraries
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyproj import Proj, transform

# good = ['']
# 2018-10-17
# 2019-01-16
# 2022-07-10

#load map plotting features
radar = pd.read_csv('J:/csv_for_2d_plotting/radar_lines.csv', index_col=0)

# Get CSV files list from a folder
path = 'J:/IceSat/FromAltimetry/'
csv_files = glob.glob(path + "*.csv")

# Read each CSV file into DataFrame
# This creates a list of dataframes
df_list = (pd.read_csv(file) for file in csv_files)

# Concatenate all DataFrames
df   = pd.concat(df_list, ignore_index=True)
df['date'] = [x[16:24] for x in df.file_name]
#remove data with low confidence
df.drop(df[df.atl06_quality_summary==1].index, inplace=True)


filenames = df.file_name.unique()
beams = df.beam.unique()

#%% Plot where there is data
#fig,ax = plt.subplots()

sns.scatterplot(data=df, x='longitude', y='latitude', hue='date')

#%%

for beam in beams:
    fg = sns.FacetGrid(data=df[df.beam==beam], hue='date', aspect=1.61)
    fg.map(plt.scatter,'longitude', 'h_li').add_legend()
    
    
#%%

fg = sns.FacetGrid(data=df, 
                   hue='date', 
                   #col= 'file_name',
                   row='beam',
                   aspect=1.61)
fg.map(plt.scatter,'longitude', 'h_li').add_legend()

#%%
for beam in beams:
    
    fig,ax = plt.subplots(2)  
    sns.scatterplot(data=df[df.beam==beam], x='longitude', y='latitude', hue='date', linewidth=0, palette = 'icefire', alpha=0.5, ax=ax[0])
    sns.scatterplot(data=df[df.beam==beam], x='longitude', y='h_li', hue='date',linewidth=0, palette = 'icefire', alpha=0.5,  ax=ax[1])

    ax[0].plot([radar.x_start,radar.x_end],[radar.y_start, radar.y_end])
    # for i in np.arange(len(radar)):
    #     ax[0].plot()




#%%

import re, seaborn as sns
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

x = df.longitude
y = df.latitude
z = df.h_li

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
plt.savefig("scatter_hue", bbox_inches='tight')