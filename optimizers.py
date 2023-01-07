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
        
        self.MC = False
        self.A = None
        self.pi = None
        
        self.local_clk = False
        self.counter = 0
        
        self.spall_gradient = False
        self.perturb_snr = 100
        self.perturb_mu = 0
        

    def _scheduler_wo_noise(self):
        pass

    def _scheduler_w_noise(self):

        self.c = self.c0*np.power(1/(1+(self.t/self.tau)), self.q)

        if self.p == 1:
            self.eps = self.eps0/(1+(self.t/self.tau))
        else:
            self.eps = self.eps0*np.power(1/(1+(self.t/self.tau)), self.p)
   
    def _scheduler_w_noise_lc(self):

        self.c = self.c0*np.power(1/(1+(self.counter/self.tau)), self.q)

        if self.p == 1:
            self.eps = self.eps0/(1+(self.counter/self.tau))
        else:
            self.eps = self.eps0*np.power(1/(1+(self.counter/self.tau)), self.p)

    def start(self, phi):
        self.phi = phi
        if self.MC:
            N = self.phi.shape[1]
            self.npaths = int(N*self.p_pick)
            self.idxs = np.arange(N)
            if self.pi is not None:
                self.idx = get_single_path(self.idxs, self.pi, N=self.npaths)
                self.idx = np.unique(self.idx)
        if self.local_clk:
            self.eps = self.eps*np.ones(self.phi.shape[1])
        if self.spall_gradient:
            self.c = 1e-12
        
    def _get_components(self):
        N = self.phi.shape[1]

        i = None
        if self.BCD and self.p_pick != 1:
            if self.MC:
                self.idx = get_next_state(self.idx, self.A, self.idxs)
                idx = self.idx
            else:
                rv = np.random.binomial(1, p=self.p_pick, size=(N, ))
                idx = rv == 1
                self.counter += idx
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
        self._get_components()
        if scheduler:
            if self.local_clk:
                self._scheduler_w_noise_lc()
            else:
                self._scheduler_w_noise()
#         print("c = {}".format(self.c))
        if self.spall_grad:
            grad = spall_gradient(df, self.phi, snr=self.snr, mu=self.mu, perturb_snr=self.perturb_snr, perturb_mu=self.perturb_mu, device=device)
#             print(grad.shape)
#             grad1 = gradient(df, self.phi, i=self.components, approx=self.approx, snr=self.snr, mu=self.mu, c=self.c, device=self.device, batch_size=self.batch_size)
#             print(grad[0, :10], grad1[0, :10])
        else:
            grad = gradient(df, self.phi, i=self.components, approx=self.approx, snr=self.snr, mu=self.mu, c=self.c, device=self.device, batch_size=self.batch_size)
            
        if self.local_clk:
            self.eps = torch.Tensor(self.eps.reshape(1, -1), device=device)
        self.phi = self.phi - self.eps*grad

        self.t += 1

class HB():
    def __init__(self, snr=np.inf, approx=1, mu_noise=0, delta=1, device="cpu", batch_size=1, seed=None, c0=1e-1, eps0=1e-3, beta=1e-4):
        self.thetha = 0
        self.phi = 0
        self.phi_prev = 0
        self.approx = approx
        self.snr = snr
        self.mu = mu_noise
        self.c = c0
        self.eps = eps0
        self.beta = beta
        self.t = 0


        self.BCD = False
        self.ncomponents = -1
        self.p_pick = delta

        self.components = None
        self.device = device

        self.batch_size = batch_size

        if seed is not None:
            torch.manual_seed(seed)
        
        self.MC = False
        self.A = None
        self.pi = None

    def _scheduler_wo_noise(self):
        pass

    def _scheduler_w_noise(self):
        pass
    
                
    def _get_components(self):
        N = self.phi.shape[1]

        i = None
        if self.BCD:
            if self.MC:
                self.idx = get_next_state(self.idx, self.A, self.idxs)
                idx = self.idx
            else:
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
        self.thetha = 0
        self.phi = phi
        self.phi_prev = torch.zeros(phi.shape, device=self.device)
        if self.MC:
            N = self.phi.shape[1]
            self.npaths = int(N*self.p_pick)
            self.idxs = np.arange(N)
            if self.pi is not None:
                self.idx = get_single_path(self.idxs, self.pi, N=self.npaths)
                self.idx = np.unique(self.idx)

    def update(self, df, scheduler=False):
        if self.p_pick == 0:
            return

        self._get_components()

        grad = gradient(df, self.phi, i=self.components, approx=self.approx, c=self.c, snr=self.snr, mu=self.mu, device=self.device, batch_size = self.batch_size)
        
        self.phi = self.phi - self.eps*grad + self.beta*(self.phi - self.phi_prev)
        self.phi_prev = self.phi
        

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
        
        self.MC = False
        self.A = None
        self.pi = None

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
            if self.MC:
                self.idx = get_next_state(self.idx, self.A, self.idxs)
                idx = self.idx
            else:
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
        if self.MC:
            N = self.phi.shape[1]
            self.npaths = int(N*self.p_pick)
            self.idxs = np.arange(N)
            if self.pi is not None:
                self.idx = get_single_path(self.idxs, self.pi, N=self.npaths)
                self.idx = np.unique(self.idx)

    def update(self, df, scheduler=False):
        if self.p_pick == 0:
            return

        self._get_components()
        
        self.p = self.phi + self.gamma*self.thetha
        grad = gradient(df, self.p, i=self.components, approx=self.approx, c=self.c, snr=self.snr, mu=self.mu, device=self.device, batch_size = self.batch_size)

#         temp = self.phi - self.eps*grad
#         self.phi = temp + self.gamma*(temp - self.thetha)
#         self.thetha = temp
        
        self.thetha = self.gamma*self.thetha - self.eps*grad
        self.phi = self.phi + self.thetha
        

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
        
        self.MC = False
        self.A = None
        self.pi = None
        
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
            if self.MC:
                self.idx = get_next_state(self.idx, self.A, self.idxs)
                idx = self.idx
            else:
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
        if self.MC:
            N = self.phi.shape[1]
            self.npaths = int(N*self.p_pick)
            self.idxs = np.arange(N)
            if self.pi is not None:
                self.idx = get_single_path(self.idxs, self.pi, N=self.npaths)
                self.idx = np.unique(self.idx)

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
            if self.MC:
                self.idx = get_next_state(self.idx, self.A, self.idxs)
                idx = self.idx
            else:
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
        if self.MC:
            N = self.phi.shape[1]
            self.npaths = int(N*self.p_pick)
            self.idxs = np.arange(N)
            if self.pi is not None:
                self.idx = get_single_path(self.idxs, self.pi, N=self.npaths)
                self.idx = np.unique(self.idx)

    def update(self, df, scheduler=False):
        if self.p_pick == 0:
            return
        self._next_gamma()
        self._get_components()

        grad = gradient(df, self.phi, i=self.components, approx=self.approx, snr=self.snr, mu=self.mu_noise, c=self.c, device=self.device, batch_size=self.batch_size)

        self.phi = self.phi + self.gamma_next*self.gamma*self.b - (1+self.gamma_next)*self.eps*grad
        self.b = self.gamma*self.b - self.eps*grad

class ADAM():
    def __init__(self, snr=np.inf, approx=1, mu_noise=0, delta=1, device="cpu", batch_size=1, seed=None, c0=1e-1, alpha=1e-2, eps0=1e-12, eps=1e-12, beta1=0.9, beta2=0.999):
        self.thetha = 0
        self.phi = 0
        self.m = 0
        self.v = 0
        self.approx = approx
        self.snr = snr
        self.mu = mu_noise
        
        self.c = c0
        self.alpha = alpha
        self.alpha_t = self.alpha
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta1_t = beta1
        self.beta2_t = beta2
        
        self.t = 0


        self.BCD = False
        self.ncomponents = -1
        self.p_pick = delta

        self.components = None
        self.device = device

        self.batch_size = batch_size

        if seed is not None:
            torch.manual_seed(seed)
        
        self.MC = False
        self.A = None
        self.pi = None

    def _next_values(self):
        
        self.alpha_t = self.alpha*np.sqrt(1 - self.beta2_t)/(1 - self.beta1_t)
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2
        
        self.t += 1
        

    def _scheduler_wo_noise(self):
        pass

    def _scheduler_w_noise(self):
        pass
    
                
    def _get_components(self):
        N = self.phi.shape[1]

        i = None
        if self.BCD:
            if self.MC:
                self.idx = get_next_state(self.idx, self.A, self.idxs)
                idx = self.idx
            else:
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
        self.thetha = 0
        self.phi = phi
        self.m = 0
        self.v = 0
        
        if self.MC:
            N = self.phi.shape[1]
            self.npaths = int(N*self.p_pick)
            self.idxs = np.arange(N)
            if self.pi is not None:
                self.idx = get_single_path(self.idxs, self.pi, N=self.npaths)
                self.idx = np.unique(self.idx)

    def update(self, df, scheduler=False):
        if self.p_pick == 0:
            return

        self._get_components()

        grad = gradient(df, self.phi, i=self.components, approx=self.approx, c=self.c, snr=self.snr, mu=self.mu, device=self.device, batch_size = self.batch_size)
        
        self.m = self.beta1*self.m + (1-self.beta1)*grad
        self.v = self.beta2*self.v + (1-self.beta2)*grad*grad

        self.phi = self.phi - self.alpha_t*self.m/(torch.sqrt(self.v) + self.eps)

        self._next_values()

class NADAM():
    def __init__(self, snr=np.inf, approx=1, mu_noise=0, delta=1, device="cpu", batch_size=1, seed=None, c0=1e-1, alpha=1e-2, eps0=1e-12, eps=1e-12, beta1=0.9, beta2=0.999):
        self.thetha = 0
        self.phi = 0
        self.m = 0
        self.v = 0
        self.approx = approx
        self.snr = snr
        self.mu = mu_noise
        
        self.c = c0
        self.alpha = alpha
        self.alpha_t = self.alpha
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta1_t = beta1
        self.beta2_t = beta2
        
        self.t = 0


        self.BCD = False
        self.ncomponents = -1
        self.p_pick = delta

        self.components = None
        self.device = device

        self.batch_size = batch_size

        if seed is not None:
            torch.manual_seed(seed)
        
        self.MC = False
        self.A = None
        self.pi = None

    def _next_values(self):
        
        self.alpha_t = self.alpha*np.sqrt(1 - self.beta2_t)/(1 - self.beta1_t)
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2
        
        self.t += 1
        

    def _scheduler_wo_noise(self):
        pass

    def _scheduler_w_noise(self):
        pass
    
                
    def _get_components(self):
        N = self.phi.shape[1]

        i = None
        if self.BCD:
            if self.MC:
                self.idx = get_next_state(self.idx, self.A, self.idxs)
                idx = self.idx
            else:
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
        self.thetha = 0
        self.phi = phi
        self.m = 0
        self.v = 0
        
        if self.MC:
            N = self.phi.shape[1]
            self.npaths = int(N*self.p_pick)
            self.idxs = np.arange(N)
            if self.pi is not None:
                self.idx = get_single_path(self.idxs, self.pi, N=self.npaths)
                self.idx = np.unique(self.idx)

    def update(self, df, scheduler=False):
        if self.p_pick == 0:
            return

        self._get_components()

        grad = gradient(df, self.phi, i=self.components, approx=self.approx, c=self.c, snr=self.snr, mu=self.mu, device=self.device, batch_size = self.batch_size)
        
        self.m = self.beta1*self.m + (1-self.beta1)*grad
        self.v = self.beta2*self.v + (1-self.beta2)*grad*grad

        self.phi = self.phi - self.alpha_t*(self.beta1*self.m + (1-self.beta1)*grad)/(torch.sqrt(self.v) + self.eps)

        self._next_values()

        
class ADAGRAD():
    def __init__(self, snr=np.inf, approx=1, mu_noise=0, delta=1, device="cpu", batch_size=1, seed=None, c0=1e-1, alpha=1e-2, eps0=1e-12, eps=1e-12):
        self.thetha = 0
        self.phi = 0
        self.G = 0
        self.approx = approx
        self.snr = snr
        self.mu = mu_noise
        
        self.c = c0
        self.alpha = alpha
        self.alpha_t = self.alpha
        self.eps = eps
        
        self.t = 0


        self.BCD = False
        self.ncomponents = -1
        self.p_pick = delta

        self.components = None
        self.device = device

        self.batch_size = batch_size

        if seed is not None:
            torch.manual_seed(seed)
        
        self.MC = False
        self.A = None
        self.pi = None
        

    def _scheduler_wo_noise(self):
        pass

    def _scheduler_w_noise(self):
        pass
    
                
    def _get_components(self):
        N = self.phi.shape[1]

        i = None
        if self.BCD:
            if self.MC:
                self.idx = get_next_state(self.idx, self.A, self.idxs)
                idx = self.idx
            else:
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
        self.thetha = 0
        self.phi = phi
        self.G = 0
        self.m = 0
        self.v = 0
        
        if self.MC:
            N = self.phi.shape[1]
            self.npaths = int(N*self.p_pick)
            self.idxs = np.arange(N)
            if self.pi is not None:
                self.idx = get_single_path(self.idxs, self.pi, N=self.npaths)
                self.idx = np.unique(self.idx)

    def update(self, df, scheduler=False):
        if self.p_pick == 0:
            return

        self._get_components()

        grad = gradient(df, self.phi, i=self.components, approx=self.approx, c=self.c, snr=self.snr, mu=self.mu, device=self.device, batch_size = self.batch_size)
        
        self.G += torch.square(grad)
        self.phi = self.phi - (self.alpha/torch.sqrt(self.G + self.eps))*grad
        
class ADADELTA():
    def __init__(self, snr=np.inf, approx=1, mu_noise=0, delta=1, device="cpu", batch_size=1, seed=None, c0=1e-1, alpha=1e-2, eps0=1e-12, eps=1e-12, gamma=0.9):
        self.thetha = 0
        self.phi = 0
        self.delta_phi = 0
        self.delta_phi_prev = 0
        self.rms_delta = 0
        self.rms_grad = 0
        self.approx = approx
        self.snr = snr
        self.mu = mu_noise
        self.gamma = gamma
        
        self.c = c0
        self.alpha = alpha
        self.alpha_t = self.alpha
        self.eps = eps
        
        self.t = 0


        self.BCD = False
        self.ncomponents = -1
        self.p_pick = delta

        self.components = None
        self.device = device

        self.batch_size = batch_size

        if seed is not None:
            torch.manual_seed(seed)
        
        self.MC = False
        self.A = None
        self.pi = None
        

    def _scheduler_wo_noise(self):
        pass

    def _scheduler_w_noise(self):
        pass
    
                
    def _get_components(self):
        N = self.phi.shape[1]

        i = None
        if self.BCD:
            if self.MC:
                self.idx = get_next_state(self.idx, self.A, self.idxs)
                idx = self.idx
            else:
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
        self.thetha = 0
        self.phi = phi
        self.delta_phi = torch.zeros(phi.shape, device=self.device)
        self.delta_phi_prev = self.delta_phi
        self.G = 0
        self.m = 0
        self.v = 0
        
        if self.MC:
            N = self.phi.shape[1]
            self.npaths = int(N*self.p_pick)
            self.idxs = np.arange(N)
            if self.pi is not None:
                self.idx = get_single_path(self.idxs, self.pi, N=self.npaths)
                self.idx = np.unique(self.idx)

    def update(self, df, scheduler=False):
        if self.p_pick == 0:
            return

        self._get_components()

        grad = gradient(df, self.phi, i=self.components, approx=self.approx, c=self.c, snr=self.snr, mu=self.mu, device=self.device, batch_size = self.batch_size)
        
        
        self.rms_delta = self.gamma*self.rms_delta + (1-self.gamma)*torch.square(self.delta_phi_prev)
        self.rms_grad = self.gamma*self.rms_grad + (1-self.gamma)*torch.square(grad)

        self.delta_phi = -1*torch.sqrt((self.eps + self.rms_delta)/(self.eps + self.rms_grad))*grad
        
        self.phi = self.phi + self.delta_phi
        
        self.delta_phi_prev = self.delta_phi
        
class RMSPROP():
    def __init__(self, snr=np.inf, approx=1, mu_noise=0, delta=1, device="cpu", batch_size=1, seed=None, c0=1e-1, alpha=1e-2, eps0=1e-12, eps=1e-12, gamma=0.9):
        self.thetha = 0
        self.phi = 0
        self.rms_grad = 0
        self.approx = approx
        self.snr = snr
        self.mu = mu_noise
        self.gamma = gamma
        
        self.c = c0
        self.alpha = alpha
        self.alpha_t = self.alpha
        self.eps = eps
        
        self.t = 0


        self.BCD = False
        self.ncomponents = -1
        self.p_pick = delta

        self.components = None
        self.device = device

        self.batch_size = batch_size

        if seed is not None:
            torch.manual_seed(seed)
        
        self.MC = False
        self.A = None
        self.pi = None
        

    def _scheduler_wo_noise(self):
        pass

    def _scheduler_w_noise(self):
        pass
    
                
    def _get_components(self):
        N = self.phi.shape[1]

        i = None
        if self.BCD:
            if self.MC:
                self.idx = get_next_state(self.idx, self.A, self.idxs)
                idx = self.idx
            else:
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
        self.thetha = 0
        self.phi = phi
        self.G = 0
        self.m = 0
        self.v = 0
        
        if self.MC:
            N = self.phi.shape[1]
            self.npaths = int(N*self.p_pick)
            self.idxs = np.arange(N)
            if self.pi is not None:
                self.idx = get_single_path(self.idxs, self.pi, N=self.npaths)
                self.idx = np.unique(self.idx)

    def update(self, df, scheduler=False):
        if self.p_pick == 0:
            return

        self._get_components()

        grad = gradient(df, self.phi, i=self.components, approx=self.approx, c=self.c, snr=self.snr, mu=self.mu, device=self.device, batch_size = self.batch_size)
        
        
        self.rms_grad = self.gamma*self.rms_grad + (1-self.gamma)*torch.square(grad)
        
        self.phi = self.phi - self.alpha*grad/torch.sqrt(self.eps + self.rms_grad)        

class ADAMAX():
    def __init__(self, snr=np.inf, approx=1, mu_noise=0, delta=1, device="cpu", batch_size=1, seed=None, c0=1e-1, alpha=1e-2, eps0=1e-12, eps=1e-12, beta1=0.9, beta2=0.999, p=1):
        self.thetha = 0
        self.phi = 0
        self.m = 0
        self.v = 0
        self.u = 0
        self.approx = approx
        self.snr = snr
        self.mu = mu_noise
        
        self.c = c0
        self.alpha = alpha
        self.alpha_t = self.alpha
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta1_t = beta1
        self.beta2_t = beta2
        
        self.t = 0


        self.BCD = False
        self.ncomponents = -1
        self.p_pick = delta

        self.components = None
        self.device = device

        self.batch_size = batch_size

        if seed is not None:
            torch.manual_seed(seed)
        
        self.MC = False
        self.A = None
        self.pi = None

    def _next_values(self):
        
        self.alpha_t = self.alpha*np.sqrt(1 - self.beta2_t)/(1 - self.beta1_t)
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2
        
        self.t += 1
        

    def _scheduler_wo_noise(self):
        pass

    def _scheduler_w_noise(self):
        pass
    
                
    def _get_components(self):
        N = self.phi.shape[1]

        i = None
        if self.BCD:
            if self.MC:
                self.idx = get_next_state(self.idx, self.A, self.idxs)
                idx = self.idx
            else:
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
        self.thetha = 0
        self.phi = phi
        self.m = 0
        self.v = 0
        
        if self.MC:
            N = self.phi.shape[1]
            self.npaths = int(N*self.p_pick)
            self.idxs = np.arange(N)
            if self.pi is not None:
                self.idx = get_single_path(self.idxs, self.pi, N=self.npaths)
                self.idx = np.unique(self.idx)

    def update(self, df, scheduler=False):
        if self.p_pick == 0:
            return

        self._get_components()

        grad = gradient(df, self.phi, i=self.components, approx=self.approx, c=self.c, snr=self.snr, mu=self.mu, device=self.device, batch_size = self.batch_size)
        
        self.m = self.beta1*self.m + (1-self.beta1)*grad
        self.v = self.beta2*self.v + (1-self.beta2)*grad*grad
        self.u = max()

        self.phi = self.phi - self.alpha_t*self.m/(torch.sqrt(self.v) + self.eps)

        self._next_values()