import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from config import *


class GD():
    def __init__(self, snr=np.inf, approx=1, mu_noise=0, delta=1, device="cpu", batch_size=1, seed=None, c0=1e-1, eps0=1e-2):
        self.phi = 0
        self.approx = approx
        self.snr = snr
        self.mu = mu_noise
        self.t = 0
        self.c0 = c0
        self.eps0 = eps0

        self.c = self.c0
        self.eps = self.eps0

        self.tau = 100
        self.q = 0.01
        self.p = 1

        self.BCD = False
        self.ncomponents = -1
        self.p_pick = delta

        self.components = None

        self.device = device
        self.batch_size = batch_size

        if seed is not None:
            torch.manual_seed(seed)

    def _scheduler_wo_noise(self):
        pass

    def _scheduler_w_noise(self):

        self.c = self.c0*np.power(1/(1+(self.t/self.tau)), self.q)

        if self.p == 1:
            self.eps = self.eps0/(1+(self.t/self.tau))
        else:
            self.eps = self.eps0*np.power(1/(1+(self.t/self.tau)), self.p)

    def start(self, phi):
        self.phi = phi

    def _get_components(self):
        N = self.phi.shape[1]

        i = None
        if self.BCD and self.p_pick != 1:
            idx = np.random.binomial(1, p=self.p_pick, size=(N, )) == 1
            i = torch.arange(N)[idx].to(self.device)
        else:
            i = None

        self.components = i
        if i is None:
            self.ncomponents = N
        else:
            self.ncomponents = i.shape[0]

    def update(self, df, scheduler=False):
        if self.p_pick == 0:
            return
        if scheduler:
            self._scheduler_w_noise()

        self._get_components()
        grad = gradient(df, self.phi, i=self.components, approx=self.approx, snr=self.snr, mu=self.mu, c=self.c, device=self.device, batch_size=self.batch_size)
        self.phi = self.phi - self.eps*grad

        self.t += 1

class NAG():
    def __init__(self, snr=np.inf, approx=1, mu_noise=0, delta=1, device="cpu", batch_size=1, seed=None, c0=1e-1, eps0=1e-3):
        self.thetha = 0
        self.phi = 0
        self.approx = approx
        self.snr = snr
        self.mu = mu_noise
        self.c = c0
        self.eps = eps0
        self.lamda = 1
        self.gamma = 0
        self.nesterov_sequence = False
        self.t = 0


        self.BCD = False
        self.ncomponents = -1
        self.p_pick = delta

        self.components = None
        self.device = device

        self.batch_size = batch_size

        if seed is not None:
            torch.manual_seed(seed)

    def _next_gamma(self):
        if self.nesterov_sequence:
            x = (1 + np.sqrt(1 + 4*np.power(self.lamda, 2)))/2
            self.gamma = (self.lamda-1)/x
            self.lamda = x
        else:
            self.gamma = 1 - 3/(self.t + 5)
        self.t += 1

    def _scheduler_wo_noise(self):
        pass

    def _scheduler_w_noise(self):
        pass
    def _get_components(self):
        N = self.phi.shape[1]

        i = None
        if self.BCD:
            idx = np.random.binomial(1, p=self.p_pick, size=(N, )) == 1
            i = torch.arange(N)[idx].to(self.device)
        else:
            i = None

        self.components = i
        if i is None:
            self.ncomponents = N
        else:
            self.ncomponents = i.shape[0]

    def start(self, phi):
        self.lamba = 1
        self.lamba_next = (1 + np.sqrt(5))/2
        self.thetha = 0
        self.phi = phi

    def update(self, df, scheduler=False):
        if self.p_pick == 0:
            return

        self._get_components()

        grad = gradient(df, self.phi, i=self.components, approx=self.approx, c=self.c, snr=self.snr, mu=self.mu, device=self.device, batch_size = self.batch_size)

        temp = self.phi - self.eps*grad
        self.phi = temp + self.gamma*(temp - self.thetha)
        self.thetha = temp

        self._next_gamma()


class NAG_sutskever():
    def __init__(self, snr=np.inf, approx=1, mu_noise=0, delta=1, device=device, batch_size=1, seed=None, c0=1e-1, eps0=1e-2):
        self.b = 0
        self.phi = 0
        self.approx = approx
        self.snr = snr
        self.mu_noise = mu_noise
        self.gamma = 1
        self.c = c0
        self.eps = eps0
        self.t = 0
        self.lamda = 0
        self.lamda_next = 1
        self.nesterov_sequence = False

        self.BCD = False
        self.ncomponents = -1
        self.p_pick = delta

        self.components = None
        self.device = device

        self.batch_size = batch_size

        if seed is not None:
            torch.manual_seed(seed)

    def _scheduler_wo_noise(self):
        pass

    def _scheduler_w_noise(self):
        pass

    def _next_gamma(self):
        if self.nesterov_sequence:
            x = (1 + np.sqrt(1 + 4*np.power(self.lamda, 2)))/2
            self.gamma = (self.lamda-1)/x
            self.lamda = x
        else:
            self.gamma = 1 - 3/(self.t + 5)
        self.t += 1

    def _get_components(self):
        N = self.phi.shape[1]

        i = None
        if self.BCD:
            idx = np.random.binomial(1, p=self.p_pick, size=(N, )) == 1
            i = torch.arange(N)[idx].to(self.device)
        else:
            i = None

        self.components = i
        if i is None:
            self.ncomponents = N
        else:
            self.ncomponents = i.shape[0]

    def start(self, phi):
        self.lamba = 1
        self.b = 0
        self.phi = phi

    def update(self, df, scheduler=False):
        if self.p_pick == 0:
            return
        self._next_gamma()

        self._get_components()

        grad = gradient(df, self.phi + self.gamma*self.b, i=self.components, approx=self.approx, c=self.c, snr=self.snr, mu=self.mu_noise, device=self.device, batch_size=self.batch_size)

        self.b = self.gamma*self.b - self.eps*grad
        self.phi = self.phi + self.b

class NAG_bengio():
    def __init__(self, snr=np.inf, approx=1, mu_noise=0, delta=1, device="cpu", batch_size=1, seed=None, c0=1e-1, eps0=1e-2):

        self.b = 0
        self.phi = 0
        self.approx = approx
        self.snr = snr
        self.gamma = 0
        self.gamma_next = 0
        self.c = c0
        self.eps = eps0
        self.mu_noise = mu_noise
        self.t = 0
        self.lamda = 1
        self.nesterov_sequence = False

        self.BCD = False
        self.ncomponents = -1
        self.p_pick = delta

        self.components = None

        self.device = device
        self.batch_size = batch_size

        if seed is not None:
            torch.manual_seed(seed)

    def _scheduler_wo_noise(self):
        pass

    def _scheduler_w_noise(self):
        pass

    def _next_gamma(self):
        if self.nesterov_sequence:

            x = (1 + np.sqrt(1 + 4*np.power(self.lamda, 2)))/2
            self.gamma = (self.lamda-1)/x
            self.lamda = x
            y = (1 + np.sqrt(1 + 4*np.power(self.lamda, 2)))/2
            self.gamma_next = (self.lamda-1)/y

        else:
            self.gamma = 1 - 3/(self.t + 5)
            self.gamma_next = 1 - 3/(self.t + 6)
        self.t += 1


    def _get_components(self):
        N = self.phi.shape[1]

        i = None
        if self.BCD:
            idx = np.random.binomial(1, p=self.p_pick, size=(N, )) == 1
            i = torch.arange(N)[idx].to(self.device)
        else:
            i = None

        self.components = i
        if i is None:
            self.ncomponents = N
        else:
            self.ncomponents = i.shape[0]

    def start(self, phi):
        self.lamba = 1
        self.b = 0
        self.phi = phi

    def update(self, df, scheduler=False):
        if self.p_pick == 0:
            return
        self._next_gamma()
        self._get_components()

        grad = gradient(df, self.phi, i=self.components, approx=self.approx, snr=self.snr, mu=self.mu_noise, c=self.c, device=self.device, batch_size=self.batch_size)

        self.phi = self.phi + self.gamma_next*self.gamma*self.b - (1+self.gamma_next)*self.eps*grad
        self.b = self.gamma*self.b - self.eps*grad
