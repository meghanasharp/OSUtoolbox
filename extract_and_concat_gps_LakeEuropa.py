
# import OS module
import os
# importing the zipfile module
from zipfile import ZipFile
#home made function from OSU-Glaciology Toolbox
# this require to have the function in the same folder
import GPS_functions as gf

#%%
#latitude_decimal_degree	longitude_decimal_degree	ellipsoidal_height_m	decimal_hour	day_of_year	year	rcvr_clk_ns

path_to_folder = 'C:/Users/trunzc/Box/SIIOS-LakeEuropa-Hiawatha-DataRepo-2022-2023/GPSdata_LakeEuropa/2022_GPSdata/Base/Base_PPP/'
path_to_save = 'C:/Users/trunzc/Box/SIIOS-LakeEuropa-Hiawatha-DataRepo-2022-2023/GPSdata_LakeEuropa/2022_GPSdata/Base/'




            
# extract csv from zipped folders and put them all in one folder

# for zipped_folder in dir_list_base:
#     if zipped_folder.endswith('_full_output.zip'):
#         print(zipped_folder)

#         with ZipFile(path_to_data + 'PPP/' + zipped_folder, mode="r") as archive:
#              for file in archive.namelist():
#                  if file.endswith(".csv"):
#                      #print('found one!')
#                      archive.extract(file, path_to_data + "PPP_csv_extracted/")


output_filename = 'GPS_concat'
df_base = gf.concat_gps_csv_files(path_to_folder = path_to_folder, 
                      path_to_save = path_to_save, 
                      output_filename = output_filename)


