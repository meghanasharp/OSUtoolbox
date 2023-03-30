
import pandas as pd
import datetime as dt
import numpy as np
# import OS module
import os
# importing the zipfile module
from zipfile import ZipFile


#path_to_CSRS_PPP_zip = 'G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/EuropaGPSdata/Base/'





#%%



def concat_gps_csv_files(path_to_CSRS_PPP_zip='', path_to_folder_to_save='', output_filename='PPP_concatenated.csv'):
    """
    This function concatenate GPS data csv files CSRS-PPP SPARK Results produced by 
    https://webapp.csrs-scrs.nrcan-rncan.gc.ca/geod/tools-outils/ppp.php?locale=en
    Files needs to be extracted from the zip format before using this function.   

    Parameters
    ----------
    path_to_folder : String
        Path to the zip folder received from gc.ca. 
    path_to_folder_to_save : String
        Path to the folder where you want to save your outputs
    output_filename : String
        Name of the file to be saved. Can also be a path with the filename

    Returns
    -------
    Saves a .csv file containing all the concatenated data.

    """
    
    #unzip files in new folder called unzipped_CSRS_PPP
    # loading the temp.zip and creating a zip object

    with ZipFile(path_to_CSRS_PPP_zip, 'r') as zObject:
    
    	# Extracting all the members of the zip
    	# into a specific location.
    	zObject.extractall(
    		path=path_to_folder_to_save + 'unzipped_CSRS_PPP')
        
    if not os.path.exists(path_to_folder_to_save + 'PPP_csv_extracted'):
       # Create a new directory because it does not exist
       os.makedirs(path_to_folder_to_save + 'PPP_csv_extracted')
    else:
        print('Folder containing extracted PPP_csv exist already in this on this path. change output path to an empty folder')
       
    for zipped_folder in path_to_folder_to_save + 'unzipped_CSRS_PPP':
        if zipped_folder.endswith('_full_output.zip'):
            print(zipped_folder)
    
            with ZipFile(path_to_folder_to_save + 'unzipped_CSRS_PPP/', mode="r") as archive:
                 for file in archive.namelist():
                     if file.endswith(".csv"):
                         #print('found one!')
                         archive.extract(file, path_to_folder_to_save + 'PPP_csv_extracted/')
    
    # # read inside each folder and find _full_output.zip zipfiles 
    # # containing the csv files with the gps data
    # #with ZipFile('unzipped_CSRS_PPP/', mode="r") as archive:
    # for zipped_folder in archive.namelist():
    #     if zipped_folder.endswith('_full_output.zip'):
    #         print(zipped_folder)
    #     #take csv file and put them all in the same folder outside of zip subfolders    
    #     with ZipFile(path_to_folder_to_save + 'unzipped_CSRS_PPP/' + zipped_folder, mode="r") as archive:
    #           for file in archive.namelist():
    #               if file.endswith(".csv"):
    #                   #print('found one!')
    #                   archive.extract(file, path_to_folder_to_save + 'PPP_csv_extracted/')


    #create dataframe with column names from the original csv files
    df = pd.DataFrame(columns=['latitude_decimal_degree', 
                               'longitude_decimal_degree', 
                               'ellipsoidal_height_m', 
                               'decimal_hour', 
                               'day_of_year', 
                               'year', 
                               'rcvr_clk_ns']) 
    #extract file names in directory
    dir_list = os.listdir('PPP_csv_extracted')
    #loop through the list of potential filenames in the folder. 
    for i,filename in enumerate(dir_list):
        whole_path = 'PPP_csv_extracted/' + filename
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
    df.to_csv( )#, encoding='utf-8-sig')

    return df

#%%
#latitude_decimal_degree	longitude_decimal_degree	ellipsoidal_height_m	decimal_hour	day_of_year	year	rcvr_clk_ns



path_to_CSRS_PPP_zip = "C:/Users/trunzc/Desktop/PPP_output_gssiradarsled.zip"     
path_to_folder_to_save ="C:/Users/trunzc/Desktop/"      
concat_gps_csv_files(path_to_CSRS_PPP_zip=path_to_CSRS_PPP_zip, path_to_folder_to_save=path_to_folder_to_save, output_filename='PPP_concatenated.csv')

                
with ZipFile(path_to_CSRS_PPP_zip, 'r') as zObject:

	# Extracting all the members of the zip
	# into a specific location.
	zObject.extractall(
		path=path_to_folder_to_save + 'unzipped_CSRS_PPP')
    
if not os.path.exists(path_to_folder_to_save + 'PPP_csv_extracted'):
   # Create a new directory because it does not exist
   os.makedirs(path_to_folder_to_save + 'PPP_csv_extracted')
else:
    print('Folder containing extracted PPP_csv exist already in this on this path. change output path to an empty folder')
   
for zipped_folder in path_to_folder_to_save + 'unzipped_CSRS_PPP':
    if zipped_folder.endswith('_full_output.zip'):
        print(zipped_folder)

        with ZipFile(path_to_folder_to_save + 'unzipped_CSRS_PPP/', mode="r") as archive:
             for file in archive.namelist():
                 if file.endswith(".csv"):
                     #print('found one!')
                     archive.extract(file, path_to_folder_to_save + 'PPP_csv_extracted/')
    




    
# #%%


# path_to_folder = path_base + 'Base_PPP_csv_extracted/'
# path_to_save = 'G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/EuropaGPSdata/'
# output_filename = 'GPS_concat_base'
# df_base = concat_gps_csv_files(path_to_folder = path_to_folder, 
#                       path_to_save = path_to_save, 
#                       output_filename = output_filename)

# #%%
# path_to_folder = path_rover + 'Rover_PPP_csv_extracted/'
# path_to_save = 'G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/EuropaGPSdata/'
# output_filename = 'GPS_concat_rover'
# df_rover = concat_gps_csv_files(path_to_folder = path_to_folder, 
#                       path_to_save = path_to_save, 
#                       output_filename = output_filename)



