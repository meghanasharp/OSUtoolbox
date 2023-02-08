
# We get annoying warnings about backends that are safe to ignore
import warnings
warnings.filterwarnings('ignore')

# Standard Python Libraries
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
#%config InlineBackend.figure_format = 'retina'

#%%

# To look through data files we use glob which is a library
# that finds all the file names matching our description
import glob
files = glob.glob('data_in/*[!.gps]')

#%%
# Impdar's loading function
from impdar.lib import load
dat = load.load_olaf.load_olaf(files)
# save the loaded data as a .mat file
dat.save('test_data_raw.mat')

# Impdar's plotting function
from impdar.lib.plot import plot_traces, plot_radargram
#%matplotlib inline
plot_radargram(dat)

