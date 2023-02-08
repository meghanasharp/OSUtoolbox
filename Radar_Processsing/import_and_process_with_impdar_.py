# # We get annoying warnings about backends that are safe to ignore
# import warnings
# warnings.filterwarnings('ignore')

# # Standard Python Libraries
# import numpy as np
# import matplotlib.p
# yplot as plt
# plt.rcParams['figure.dpi'] = 300
# #%config InlineBackend.figure_format = 'retina'

# #%%

# # To look through data files we use glob which is a library
# # that finds all the file names matching our description
# import glob
# path =' G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/Radar/LakeEuropa2022_RadarData_mat_files/corrected_elevation'
# files = glob.glob(path + '/*[!.gps]')

# #%%
# # Impdar's loading function
# from impdar.lib import load
# dat = load.load_olaf.load_olaf(files)
# # save the loaded data as a .mat file
# dat.save('test_data_raw.mat')

# # Impdar's plotting function
# from impdar.lib.plot import plot_traces, plot_radargram
# #%matplotlib inline
# plot_radargram(dat)


import functions as fc
from impdar import load
path = ' G:/Shared drives/6 Greenland Europa Hiawatha Projects/Lake Europa/Radar/LakeEuropa2022_RadarData_mat_files/corrected_elevation'
file = load.load('mat',path)
MHz = 20

fc.file_process(file, MHz, migration=False, save=False)