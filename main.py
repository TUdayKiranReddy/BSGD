import torch
torch.set_default_dtype(torch.float32)

import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

from utils import *
from optimizers import *
from config import *

output_plots_dir = 'quad_logsumexp_plots'
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
        print(output_plots_dir + " already exists!")

#########################################################
############## BATCH COORDINATE DESCENT #################
#########################################################
## NOISELESS APPROX GRADIENTS CONFIGURATION

snr = np.inf
approx = 1
mu_noise = 0
batch_size = 512
delta = 0.2
seed = 69
isDNN = True
is_BCD = False
#######################################################

values_gd_approx = simulate(f, f, GD, approx=approx, mu_noise=mu_noise, snr=snr, batch_size=batch_size, is_BCD=is_BCD, delta=delta, seed=seed, isDNN=isDNN)
values_nag_approx = simulate(f, f, NAG, approx=approx, mu_noise=mu_noise, snr=snr, batch_size=batch_size, is_BCD=is_BCD, delta=delta, seed=seed, isDNN=isDNN)
values_nag_suts_approx = simulate(f, f, NAG_sutskever, approx=approx, mu_noise=mu_noise, snr=snr, batch_size=batch_size, is_BCD=is_BCD, delta=delta, seed=seed, isDNN=isDNN)
values_nag_ben_approx = simulate(f, f, NAG_bengio, approx=approx, mu_noise=mu_noise, snr=snr, batch_size=batch_size, is_BCD=is_BCD, delta=delta, seed=seed, isDNN=isDNN)
######################################################

## 50dB-NOISY APPROX GRADIENTS CONFIGURATION

snr_approx = 10
approx = 1
mu_noise = 0
batch_size = 512
seed = 70
#####################################################

values_gd_approx_noisy = simulate(f, f, GD, approx=approx, mu_noise=mu_noise, snr=snr_approx, batch_size=batch_size, is_BCD=is_BCD, delta=delta, seed=seed, isDNN=isDNN)
values_nag_approx_noisy = simulate(f, f, NAG, approx=approx, mu_noise=mu_noise, snr=snr_approx, batch_size=batch_size, is_BCD=is_BCD, delta=delta, seed=seed, isDNN=isDNN)
values_nag_suts_approx_noisy = simulate(f, f, NAG_sutskever, approx=approx, mu_noise=mu_noise, snr=snr_approx, batch_size=batch_size, is_BCD=is_BCD, delta=delta, seed=seed, isDNN=isDNN)
values_nag_ben_approx_noisy = simulate(f, f, NAG_bengio, approx=approx, mu_noise=mu_noise, snr=snr_approx, batch_size=batch_size, is_BCD=is_BCD, delta=delta, seed=seed, isDNN=isDNN)

#### PLOTTING #################################################
plt.figure(figsize=(20, 10))
plt.suptitle(r"Comparing BSGD, $\rho$={}".format(delta), fontsize=30)

plt.subplot(1,2,1)
plt.title(r"Approx Gradients at SNR=$\infty$ dB", fontsize=25)
plt.semilogy(values_nag_approx, label="NAG")
plt.semilogy(values_nag_suts_approx, '-.', label="Sutskever's NAG")
plt.semilogy(values_nag_ben_approx, '--', label="Bengio's NAG")
plt.semilogy(values_gd_approx, label="BSGD")
plt.xlabel(r't', fontsize=20)
plt.ylabel(r'$|J({\boldsymbol{\theta}_t}) - J({\boldsymbol{\theta}^*})|$', fontsize=20)
plt.xticks(fontsize=14);
plt.yticks(fontsize=14);


plt.subplot(1,2,2)
plt.title("Approx Gradients at SNR={} dB".format(snr_approx), fontsize=25)
plt.semilogy(values_nag_approx_noisy, label="NAG")
plt.semilogy(values_nag_suts_approx_noisy, '-.', label="Sutskever's NAG")
plt.semilogy(values_nag_ben_approx_noisy, '--', label="Bengio's NAG")
plt.semilogy(values_gd_approx_noisy, label="BSGD")
plt.xlabel(r't', fontsize=20)
plt.ylabel(r'$|J({\boldsymbol{\theta}_t}) - J({\boldsymbol{\theta}^*})|$', fontsize=20)
plt.xticks(fontsize=14);
plt.yticks(fontsize=14);

plt.legend(loc='upper right', bbox_to_anchor=(0.85, -0.08),
          fancybox=True, shadow=True, ncol=5, fontsize=25)

plt.savefig('./'+output_plots_dir+'/asynchronous_comparisions_all_{}dB.jpg'.format(snr_approx), bbox_inches='tight')


if saveData:
    np.save('./'+output_plots_dir+'/data_files' + '/values_gd_approx.npy', values_gd_approx)
    np.save('./'+output_plots_dir+'/data_files' + '/values_nag_approx.npy', values_nag_approx)
    np.save('./'+output_plots_dir+'/data_files' + '/values_nag_suts_approx.npy', values_nag_suts_approx)
    np.save('./'+output_plots_dir+'/data_files' + '/values_nag_ben_approx.npy', values_nag_ben_approx)

    np.save('./'+output_plots_dir+'/data_files' + '/values_gd_approx_noisy.npy', values_gd_approx_noisy)
    np.save('./'+output_plots_dir+'/data_files' + '/values_nag_approx_noisy.npy', values_nag_approx_noisy)
    np.save('./'+output_plots_dir+'/data_files' + '/values_nag_suts_approx_noisy.npy', values_nag_suts_approx_noisy)
    np.save('./'+output_plots_dir+'/data_files' + '/values_nag_ben_approx_noisy.npy', values_nag_ben_approx_noisy)


########################################################################################
######################## BSGD varying \rho #####################

approx = 1
mu_noise = 0
batch_size = 128
deltas = np.linspace(0.05, 0.2, 11)
'''
N = 100
A = torch.Tensor(np.load('data_files/A_10.npy')).to(device)

def f(x, A=A, device="cpu"):
    y = x@A@x.T
    return torch.diag(y)

def df(x, A=A, device=device):
    return (2*x@A).to(device)
'''


phi = 1e-2*torch.ones(size=(1, N)).to(device)

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
        snr_approx = 50
        Values_gd_approx_noisy.append(simulate(f, f, GD, approx=approx, mu_noise=mu_noise, snr=snr_approx, batch_size=batch_size, is_BCD=True, delta=delta, scheduler=scheduler, seed=seeds[j], ITR_LIM=ITR_LIM, load_phi=False, phi=phi))

    values_gd_approx += np.array(Values_gd_approx)
    values_gd_approx_noisy += np.array(Values_gd_approx_noisy)

values_gd_approx /= nMC
values_gd_approx_noisy /= nMC

plot_var_delta(values_gd_approx, values_gd_approx_noisy, deltas, title=r"BSGD varying $\rho$", snr=snr_approx, savepath="./"+output_plots_dir+"/varying_rho_{}.jpeg".format(snr_approx))

if saveData:
    np.save('./'+output_plots_dir+'/data_files' + '/Values_gd_approx.npy', values_gd_approx)
    np.save('./'+output_plots_dir+'/data_files' + '/Values_gd_approx_noisy.npy', values_gd_approx_noisy)
##################################################################################
#################### Constant vs Blum ###########################
# from config import *
################ Config #############
approx = 1
mu_noise = 0
batch_size = 512
ITR_LIM = int(1e4)
step = 1
seed = 69
load_phi = False
c = 1e-6
is_BCD = True
delta = 0.2
return_params = True
tau = 2e2
(p, q) = (1, 0.02)

phi = 1e-1*torch.ones(size=(1, N)).to(device)

snr = -10

x_gd_approx, values_gd_approx,\
params_c_gd_approx, params_eps_gd_approx = simulate(f, f, GD, is_require_coords=True, approx=approx,
                                                    mu_noise=mu_noise, snr=snr, batch_size=batch_size,
                                                    scheduler=False, ITR_LIM=ITR_LIM, step=step, seed=seed,
                                                    load_phi=load_phi, return_params=return_params,
                                                    tau=tau, phi=phi, p=p, q=q, is_BCD=is_BCD, delta=delta, c=c)
x_gd_approx_blum, values_gd_approx_blum,\
params_c_gd_approx_blum, params_eps_gd_approx_blum = simulate(f, f, GD, is_require_coords=True,
                                                              approx=approx, mu_noise=mu_noise, snr=snr,
                                                              batch_size=batch_size, scheduler=True,
                                                              ITR_LIM=ITR_LIM, step=step, seed=seed,
                                                              load_phi=load_phi, return_params=return_params,
                                                              tau=tau, phi=phi, p=p, q=q, is_BCD=is_BCD, delta=delta, c=c)


snr = 0

x_gd_approx, values_gd_approx_noisy,\
params_c_gd_approx, params_eps_gd_approx = simulate(f, f, GD, is_require_coords=True, approx=approx,
                                                    mu_noise=mu_noise, snr=snr, batch_size=batch_size,
                                                    scheduler=False, ITR_LIM=ITR_LIM, step=step, seed=seed,
                                                    load_phi=load_phi, return_params=return_params,
                                                    tau=tau, phi=phi, p=p, q=q, is_BCD=is_BCD, delta=delta, c=c)
x_gd_approx_blum, values_gd_approx_blum_noisy,\
params_c_gd_approx_blum, params_eps_gd_approx_blum = simulate(f, f, GD, is_require_coords=True,
                                                              approx=approx, mu_noise=mu_noise, snr=snr,
                                                              batch_size=batch_size, scheduler=True,
                                                              ITR_LIM=ITR_LIM, step=step, seed=seed,
                                                              load_phi=load_phi, return_params=return_params,
                                                              tau=tau, phi=phi, p=p, q=q, is_BCD=is_BCD, delta=delta, c=c)


plt.figure(figsize=(12, 8))
plt.title(r"Constant step size vs Blum's condition, $\rho$ = 0.2", fontsize=25)

plt.semilogy(values_gd_approx_noisy, label="Constant Step Size at SNR = 10 dB")
plt.semilogy(values_gd_approx, label="Constant Step Size at SNR = 20 dB")
plt.semilogy(values_gd_approx_blum_noisy, label="Blum's conditions at SNR = 10 dB")
plt.semilogy(values_gd_approx_blum, label="Blum's conditions at SNR = 20 dB")
plt.xlabel(r't', fontsize=20)
plt.ylabel(r'$|J({\boldsymbol {\theta}_t}) - J({\boldsymbol {\theta}^*})|$', fontsize=20)
plt.xticks(fontsize=14);
plt.yticks(fontsize=14);

plt.legend(loc='upper right', bbox_to_anchor=(1.06, -0.08),
          fancybox=True, shadow=True, ncol=2, fontsize=20)
plt.savefig('./'+output_plots_dir+'/constant_v_blum_together.jpg', bbox_inches='tight')

if saveData:
    np.save('./'+output_plots_dir+'/data_files' + '/values_gd_approx.npy', values_gd_approx)
    np.save('./'+output_plots_dir+'/data_files' + '/values_gd_approx_noisy.npy', values_gd_approx_noisy)
    np.save('./'+output_plots_dir+'/data_files' + '/values_gd_approx_blum.npy', values_gd_approx_blum)
    np.save('./'+output_plots_dir+'/data_files' + '/Values_gd_approx_blum_noisy.npy', values_gd_approx_blum_noisy)
