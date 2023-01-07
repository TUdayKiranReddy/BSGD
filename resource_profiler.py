import torch
torch.set_default_dtype(torch.float32)

import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

from utils import *
from optimizers import *
from config import *

output_plots_dir = 'resource_profiler_final_2'
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


memory_prof = True
peak = True

if memory_prof:
    from memory_profiler import memory_usage
else:
    import time


approx = 1
mu_noise = 0
batch_size = 32
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
ITR_LIM = 100

nMC = 5
scheduler = True

seeds = 60 + np.arange(nMC)
################################

exact_mem = 0
approx_mem = 0
exact_cpu = 0
approx_cpu = 0


def func(snr_g, delta):
    return simulate(f, f, GD, approx=approx, mu_noise=mu_noise, snr=snr_g, batch_size=batch_size, is_BCD=True, delta=delta, scheduler=scheduler, seed=seeds[j], ITR_LIM=int(ITR_LIM/delta), load_phi=False, phi=phi)

def gf(a, n=100):
    import time
    time.sleep(2)
    b = [a] * n
    time.sleep(1)
    return b
for j in range(nMC):
    Exact_mem = []
    Exact_cpu = []
    Approx_mem = []
    Approx_cpu = []
    for delta in deltas:

        print("Delta = ", delta)
        # Exact
        snr = np.inf
        if memory_prof:
            if peak:
                m = np.max(memory_usage((func, (snr, delta, ))))
            else:
                m = np.mean(memory_usage((func, (snr, delta, ))))
            Exact_mem.append(m)
        else:
            t0 = time.time()
            simulate(f, f, GD, approx=approx, mu_noise=mu_noise, snr=snr, batch_size=batch_size, is_BCD=True, delta=delta, scheduler=scheduler, seed=seeds[j], ITR_LIM=int(ITR_LIM/delta), load_phi=False, phi=phi)
            Exact_cpu.append(time.time()-t0)
        print("Yo")

        # Approximate Noisy
        snr_approx = 50

        if memory_prof:
            if peak:
                m = np.max(memory_usage((func, (snr_approx, delta, ))))
            else:
                m = np.mean(memory_usage((func, (snr_approx, delta, ))))
            Approx_mem.append(m)
        else:
            t0 = time.time()
            simulate(f, f, GD, approx=approx, mu_noise=mu_noise, snr=snr_approx, batch_size=batch_size, is_BCD=True, delta=delta, scheduler=scheduler, seed=seeds[j], ITR_LIM=int(ITR_LIM/delta), load_phi=False, phi=phi)
            Approx_cpu.append(time.time()-t0)

    exact_mem += np.array(Exact_mem)
    exact_cpu += np.array(Exact_cpu)
    approx_mem += np.array(Approx_mem)
    approx_cpu += np.array(Approx_cpu)

exact_mem /= nMC
exact_cpu /= nMC
approx_cpu /= nMC
approx_mem /= nMC

plt.figure()
if memory_prof:
    plt.title('Memory consumption')
    plt.plot(deltas, exact_mem, label='Exact')
    plt.plot(deltas, approx_mem, label='Aprrox')
else:
    plt.title('Time consumption')
    plt.plot(deltas, exact_cpu, label='Exact')
    plt.plot(deltas, approx_cpu, label='Approx')

plt.legend()
plt.show()

if saveData:
    if memory_prof:
        np.save('./'+output_plots_dir+'/data_files' + '/exact_memory.npy', exact_mem)
        np.save('./'+output_plots_dir+'/data_files' + '/approx_memory.npy', approx_mem)
    else:
        np.save('./'+output_plots_dir+'/data_files' + '/exact_cpu.npy', exact_cpu)
        np.save('./'+output_plots_dir+'/data_files' + '/approx_cpu.npy', approx_cpu)
