# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 14:57:45 2022

@author: trunzc
"""


import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
# import OS module
import os


# to unzip all in gitbash:  unzip "*.zip"

path_to_folder = 'J:/IceSat/csv_from_openaltimetry/'

column_photon = ['latitude', 'longitude', 'photon height', 'confidence code']

column_elev = ['segment_id', 'longitude','latitude','h_li','atl06_quality_summary','track_id','beam','file_name']



# df = pd.DataFrame(columns=column_elev)#columns=['latitude_decimal_degree']) 

# dir_list = os.listdir(path_to_folder)
dir_list = os.listdir(path_to_folder)
#%%
df_elev = []
df_elev = pd.DataFrame(columns=column_elev)#columns=['latitude_decimal_degree']) 
for i,filename in enumerate(dir_list):
    whole_path = path_to_folder + filename
    if os.path.exists(whole_path):
        if 'csv' in filename: #
            if 'elev' in filename:
                print('concat:', filename)
                data = pd.read_csv(whole_path)
                #df.append(data)
                df_elev = pd.concat([df_elev,data])
            else:
                print('pass:', filename)
        else:
            print('pass:', filename)
    else:
        print('path does not exist')
    #make index from 0 to n, without repetition
df_elev = df_elev.reset_index(drop=True)

# #%%
#del df_photon
df_photon = []
df_photon = pd.DataFrame(columns=column_photon)#columns=['latitude_decimal_degree']) 
for i,filename in enumerate(dir_list):
    whole_path = path_to_folder + filename
    if os.path.exists(whole_path):
        if 'csv' in filename: #
            if 'photon' in filename:
                print('concat:', filename)
                data = pd.read_csv(whole_path)
                data['filename']=filename
                #df.append(data)
                df_photon = pd.concat([df_photon,data])
            else:
                print('pass:', filename)
        else:
            print('pass:', filename)
    else:
        print('path does not exist')
    #make index from 0 to n, without repetition
df_photon = df_photon.reset_index(drop=True)

#%%
filenames = df_photon.filename.unique()
fig,ax = plt.subplots(2)
for filename in filenames:
    selection = df_photon['filename']==filename
    ax[0].plot(df_photon[selection].longitude, df_photon[selection].latitude,'.')
    ax[1].plot(df_photon[selection].longitude, df_photon[selection]['photon height'], '.')
ax[0].legend(filenames)

#%%
# filenames = df.file_name.unique()

# fig,ax = plt.subplots(2)


# for filename in filenames:
#     selection = (df['file_name']==filename) | (df['beam']=='gt3r')
#     ax[0].plot(df[selection].longitude, df[selection].latitude,'.')
#     ax[1].plot(df[selection].longitude, df[selection].h_li, '.')
# ax[0].legend(filenames)



















