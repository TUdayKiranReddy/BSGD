import torch
torch.set_default_dtype(torch.float32)

import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

from utils import *
from optimizers import *
from config import *

output_plots_dir = 'phase_change'
saveData = True
import os
try:
    os.mkdir('./'+output_plots_dir)
except OSError as error:
    print(output_plots_dir + " already exists!")

if saveData:
    try:
        os.mkdir('./'+output_plots_dir+'/data_files')
    except OSError as error:
        print(output_plots_dir + '/data_files' + " already exists!")

########################################################################################
######################## BSGD varying \rho #####################

approx = 1
mu_noise = 0
batch_size = 128
deltas = np.linspace(0.1, 1, 10)

'''
N = 100
A = torch.Tensor(np.load('data_files/A_10.npy')).to(device)

def f(x, A=A, device="cpu"):
    y = x@A@x.T
    return torch.diag(y)

def df(x, A=A, device=device):
    return (2*x@A).to(device)
'''

phi = torch.ones(size=(1, N)).to(device)

###############################
# Approximate
ITR_LIM = 1000

nMC = 5
scheduler = True

seeds = 60 + np.arange(nMC)
################################
values_gd_approx = 0
values_gd_approx_noisy = 0

for j in range(nMC):
    Values_gd_approx = []
    Values_gd_approx_noisy = []

    for delta in deltas:
        print("Delta = ", delta)
        # Exact
        snr = np.inf
        Values_gd_approx.append(simulate(f, f, GD, approx=approx, mu_noise=mu_noise, snr=snr, batch_size=batch_size, is_BCD=True, delta=delta, scheduler=scheduler, seed=seeds[j], ITR_LIM=ITR_LIM, load_phi=False, phi=phi))

        # Approximate Noisy
        snr_approx = 0
        Values_gd_approx_noisy.append(simulate(f, f, GD, approx=approx, mu_noise=mu_noise, snr=snr_approx, batch_size=batch_size, is_BCD=True, delta=delta, scheduler=scheduler, seed=seeds[j], ITR_LIM=ITR_LIM, load_phi=False, phi=phi))

    values_gd_approx += np.array(Values_gd_approx)
    values_gd_approx_noisy += np.array(Values_gd_approx_noisy)

values_gd_approx /= nMC
values_gd_approx_noisy /= nMC

plot_var_delta(values_gd_approx, values_gd_approx_noisy, deltas, title=r"BSGD varying $\rho$", snr=snr_approx, savepath="./"+output_plots_dir+"/varying_rho_{}.jpeg".format(snr_approx))

if saveData:
    np.save('./'+output_plots_dir+'/data_files' + '/Values_gd_approx.npy', values_gd_approx)
    np.save('./'+output_plots_dir+'/data_files' + '/Values_gd_approx_noisy.npy', values_gd_approx_noisy)

