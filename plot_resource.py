import torch
torch.set_default_dtype(torch.float32)

import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


PATH = 'resource_profiler_final_2/data_files/'

deltas = np.linspace(0.1, 1, 10)

exact_cpu = np.load(PATH + 'exact_cpu.npy')
exact_mem = np.load(PATH + 'exact_memory.npy')
approx_cpu = np.load(PATH + 'approx_cpu.npy')
approx_mem = np.load(PATH + 'approx_memory.npy')

fig, ax = plt.subplots()
fig.set_figheight(6)
fig.set_figwidth(8)

ax.set_title('Resource tradeoff in BSGD', fontsize=25)
ax.plot(deltas, approx_cpu, 'r.-',label='Approx Gradients')
ax.plot(deltas, exact_cpu, 'g.-',label='Exact Gradients')
ax.set_ylabel('Time (in s)', fontsize=20)
ax.set_xlabel(r'$\rho$', fontsize=20)
ax.set_ylim([4, 7])

ax2 = ax.twinx()
ax2.plot(deltas, approx_mem, 'r^-', alpha=0.5)
ax2.plot(deltas, exact_mem, 'g^-', alpha=0.5)
ax2.set_ylabel('Memory (in MB)', fontsize=20)
ax2.set_ylim([273.2, 274])
fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.04), ncol=2, fontsize=15)


plt.savefig(PATH + '/plot_1.jpeg', bbox_inches='tight')
plt.show()
