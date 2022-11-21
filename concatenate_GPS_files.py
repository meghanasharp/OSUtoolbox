
import pandas as pd
import datetime as dt
import numpy as np
# import OS module
import os
# importing the zipfile module
from zipfile import ZipFile



def concat_gps_csv_files(path_to_folder=None, path_to_save=None, output_filename=None):
    """
    This function concatenate GPS data csv files CSRS-PPP SPARK Results produced by 
    https://webapp.csrs-scrs.nrcan-rncan.gc.ca/geod/tools-outils/ppp.php?locale=en
    Files needs to be extracted from the zip format before using this function.   

    Parameters
    ----------
    path_to_folder : String
        Full path to the folder containing the csv files to concatenate.
        The folder should only contain the csv files. 
    path_to_save : String
        Full path to the folder where the final concatenated csv will be saved.
    output_filename : String
        Name of the file to be saved.

    Returns
    -------
    Saves a .csv file containing all the concatenated data.

    """

    #create dataframe with column names from the original csv files
    df = pd.DataFrame(columns=['latitude_decimal_degree', 
                               'longitude_decimal_degree', 
                               'ellipsoidal_height_m', 
                               'decimal_hour', 
                               'day_of_year', 
                               'year', 
                               'rcvr_clk_ns']) 
    #extract file names in directory
    dir_list = os.listdir(path_to_folder)
    #loop through the list of potential filenames in the folder. 
    for i,filename in enumerate(dir_list):
        whole_path = path_to_folder + filename
        if os.path.exists(whole_path):
            data = pd.read_csv(whole_path)
            #df.append(data)
            df = pd.concat([df,data])
        else:
            print('path does not exist')
    #make index from 0 to n, without repetition
    df = df.reset_index(drop=True)
    
    #create time array in local and utc time    
    df['time_utc'] = dt.datetime.strptime("01/01/22", "%m/%d/%y") + pd.to_timedelta(df.day_of_year-1,'d')  + pd.to_timedelta(df.decimal_hour,'h') #do we need to substract one day?? ask christian
    df['time_utc'] = np.round(df['time_utc'].astype(np.int64), -9).astype('datetime64[ns]')
    df['time_local'] = df['time_utc'] + dt.timedelta(hours = -3)
    
    #save to a csv file
    df.to_csv(path_to_save + output_filename + '.csv')#, encoding='utf-8-sig')

    return df

#%%
#latitude_decimal_degree	longitude_decimal_degree	ellipsoidal_height_m	decimal_hour	day_of_year	year	rcvr_clk_ns

path_base = 'G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/EuropaGPSdata/Base/'
path_rover = 'G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/EuropaGPSdata/Rover/'


dir_list_base = os.listdir(path_base + 'Base_PPP')
dir_list_rover = os.listdir(path_rover + 'Rover_PPP')

            
# extract csv from zipped folders 

for zipped_folder in dir_list_base:
    if zipped_folder.endswith('_full_output.zip'):
        print(zipped_folder)

        with ZipFile(path_base + 'Base_PPP/' + zipped_folder, mode="r") as archive:
             for file in archive.namelist():
                 if file.endswith(".csv"):
                     #print('found one!')
                     archive.extract(file, path_base + "Base_PPP_csv_extracted/")


for zipped_folder in dir_list_rover:
    if zipped_folder.endswith('_full_output.zip'):
        print(zipped_folder)

        with ZipFile(path_rover + 'Rover_PPP/' + zipped_folder, mode="r") as archive:
             for file in archive.namelist():
                 if file.endswith(".csv"):
                     #print('found one!')
                     archive.extract(file, path_rover + "Rover_PPP_csv_extracted/")

                





    
#%%


path_to_folder = path_base + 'Base_PPP_csv_extracted/'
path_to_save = 'G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/EuropaGPSdata/'
output_filename = 'GPS_concat_base'
df_base = concat_gps_csv_files(path_to_folder = path_to_folder, 
                      path_to_save = path_to_save, 
                      output_filename = output_filename)

#%%
path_to_folder = path_rover + 'Rover_PPP_csv_extracted/'
path_to_save = 'G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/EuropaGPSdata/'
output_filename = 'GPS_concat_rover'
df_rover = concat_gps_csv_files(path_to_folder = path_to_folder, 
                      path_to_save = path_to_save, 
                      output_filename = output_filename)



