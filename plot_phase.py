import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

from utils import *
from optimizers import *
from config import *

output_plots_dir = 'phase_change'
PATH = 'phase_change/data_files/'

values_gd_approx = np.load(PATH + 'Values_gd_approx.npy')
values_gd_approx_noisy = np.load(PATH + 'Values_gd_approx_noisy.npy')
snr_approx = 50
deltas = np.arange(0.05, 1, 100)

#print(values_gd_approx[50:60].shape)
#print(Values_gd_approx_noisy.shape)
plot_var_delta(values_gd_approx, values_gd_approx_noisy, deltas, title=r"BSGD varying $\rho$", snr=snr_approx, savepath="./"+output_plots_dir+"/varying_rho_{}_1.jpeg".format(snr_approx))
