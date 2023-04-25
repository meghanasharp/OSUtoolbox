
import pandas as pd
import datetime as dt
import numpy as np
# import OS module
import os
# importing the zipfile module
from zipfile import ZipFile



def concat_gps_csv_files(path_to_folder=None, path_to_save=None, output_filename='extracted_ppp'):
    """
    This function concatenate GPS data csv files CSRS-PPP SPARK Results produced by 
    https://webapp.csrs-scrs.nrcan-rncan.gc.ca/geod/tools-outils/ppp.php?locale=en
    Files needs to be extracted from the zip format before using this function.   

    Parameters
    ----------
    path_to_folder : String
        Full path to the folder containing the csv files to concatenate.
        The zipped folder received from CSRS-PPP should be unzipped. but leave the subfolder zipped. 
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
    
    
    
    for zipped_folder in dir_list:
        #select only the files that are going to contain data
        if zipped_folder.endswith('_full_output.zip'):
            print('opening zipped folder :', zipped_folder)
            archive = ZipFile(path_to_folder + zipped_folder) 
#            with ZipFile(path_to_folder + zipped_folder, mode="r") as archive:
            
            for file in archive.namelist():
                if file.endswith(".csv"):
                    data = pd.read_csv(archive.open(file))
                    #print('found one!')
                    #archive.extract(file, path_to_data + "PPP_csv_extracted/")

                    #df.append(data)
                    df = pd.concat([df,data])

    #make index from 0 to n, without repetition
    df = df.reset_index(drop=True)
    
    #create time array in local and utc time    
    df['time_utc'] = dt.datetime.strptime("01/01/22", "%m/%d/%y") + pd.to_timedelta(df.day_of_year-1,'d')  + pd.to_timedelta(df.decimal_hour,'h') #do we need to substract one day?? ask christian
    df['time_utc'] = np.round(df['time_utc'].astype(np.int64), -9).astype('datetime64[ns]')
    df['time_local'] = df['time_utc'] + dt.timedelta(hours = -3)
    
    #save to a csv file
    df.to_csv(path_to_save + output_filename + '.csv')#, encoding='utf-8-sig')

    return df





