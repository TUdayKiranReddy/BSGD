import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import *

def first_order_approx(f, phi, i, batch_size=64, eps=1e-6, snr=10, mu=0, device=device):
    n = phi.shape[0]
    e_i = torch.zeros((batch_size, n)).to(device)
    e_i[i] = 1

    df = (add_noise(f(phi + eps*e_i), snr, mu, device) - add_noise(f(phi - eps*e_i), snr, mu, device))
    return df/(2*eps)


def add_noise(y, snr, mu=0, device=device, constant_snr=True):
    if snr == np.inf:
        return y

    if constant_snr:
        std = torch.pow(torch.tensor(10).to(device), -1*snr/10)
    else:
        sig_p = torch.pow(y, 2).to(device)
        sig_n = sig_p*torch.pow(torch.tensor(10).to(device), -1*snr/20)
        std = torch.clamp(torch.sqrt(sig_n), max=1e11)

    mean = mu*torch.ones(y.shape).to(device)
    noise = mean + std*(torch.randn(y.shape).to(device))
    y_noise = y + noise
    return y_noise

def first_order_approx(f, phi, i, batch_size=64, eps=1e-6, snr=10, mu=0, device=device):
    n = phi.shape[1]
    batch_phi = torch.squeeze(phi.expand(batch_size, 1, n), dim=1)

    e_i = torch.zeros((batch_size, n)).to(device)
    
    eps_batch = torch.zeros((batch_size, 1)).to(device)
    
    for b_i, ii in enumerate(i):
        e_i[b_i, ii] = 1
        if type(eps) == np.ndarray: 
            eps_batch[b_i, 0] = eps[ii]
        else:
            eps_batch[b_i, 0] = eps
    
    df = (add_noise(f(batch_phi + eps_batch*e_i), snr, mu, device) - add_noise(f(batch_phi - eps_batch*e_i), snr, mu, device))
    return df/(2*eps_batch[:,0])

# https://doi.org/10.1016/S0005-1098(96)00149-5
def spall_gradient(df, phi, c=1e-1, snr=np.inf, mu=0, perturb_snr=100, perturb_mu=0, device=device, p_pick=0.5):
#     print("c = {}".format(c))
#     v = perturb_mu*torch.ones(phi.shape).to(device) + np.power(10, -1*perturb_snr/20)*(torch.randn(phi.shape).to(device))
    v = torch.Tensor(2*np.random.binomial(1, p=p_pick, size=phi.shape) - 1).to(device)
    f_n = add_noise(df(phi + c*v), snr, mu, device) - add_noise(df(phi), snr, mu, device)
#     f_n = df(phi + c*v)
#     print(phi[0, :10], v[0, :10], (c*v[0, :10]), f_n)
#     print(v.shape)
    return f_n/(c*v)

def gradient(df, phi, i=None, approx=None, snr=np.inf, mu=0, c=1e-6, batch_size=64, device=device):

    if approx is None:
        gd = add_noise(df(phi), snr, mu, device)
        if i is None:
            return gd
        else:
            mask = np.logical_not(np.isin(np.arange(phi.shape[1]), np.array(i)))
            gd[:, mask] = 0
            return gd

    n = phi.shape[1]
    grad = torch.zeros(size=(1, n)).to(device)

    if i is None:
        n_batches = int(n//batch_size)
        left_idxs = int(n - n_batches*batch_size)
        iidxs = torch.arange(n)

        for i in range(n_batches):
            grad[0, (i*batch_size):((i+1)*batch_size)] = first_order_approx(df, phi, iidxs[(i*batch_size):((i+1)*batch_size)], batch_size=batch_size, eps=c, snr=snr, mu=mu, device=device)
        if left_idxs != 0:
            #print(phi.shape, first_order_approx(df, phi, iidxs[(n_batches*batch_size):], batch_size=left_idxs, eps=c, snr=snr, mu=mu, device=device))
            grad[0, (n_batches*batch_size):] = first_order_approx(df, phi, iidxs[(n_batches*batch_size):], batch_size=left_idxs, eps=c, snr=snr, mu=mu, device=device)
    else:
        len_i = len(i)
        n_batches = int(len_i//batch_size)
        left_idxs = int(len_i - n_batches*batch_size)

        for ii in range(n_batches):
            grad[0, (ii*batch_size):((ii+1)*batch_size)] = first_order_approx(df, phi, i[(ii*batch_size):((ii+1)*batch_size)], batch_size=batch_size, eps=c, snr=snr, mu=mu, device=device)

        if left_idxs != 0:
            grad[0, i[(n_batches*batch_size):]] = first_order_approx(df, phi, i[(n_batches*batch_size):], batch_size=left_idxs, eps=c, snr=snr, mu=mu, device=device)

    return grad


def simulate(grad, f, GD, approx=1, mu_noise=0, snr=np.inf, is_BCD=False, delta=0.5, is_require_coords=False, batch_size=1, scheduler=True, ITR_LIM=10000, step=1, seed=None, load_phi=False, return_params=False, tau=2e2, phi=None, p=1, q=0.02, c=1e-1, N=N, isDNN=isDNN, opt_f=opt_f, eps=1e-2, MC=False, A_trans=None, pi=None, local_clock=False, spall_grad=False):
    if load_phi:
        phi = torch.Tensor(np.load('./data_files/phi_2.npy').reshape(1, -1)).to(device)
    elif phi is None:
        scale = 1e-3 if N == 1000 else 1.0
        phi = scale*torch.ones(size=(1, N)).to(device)

    if isDNN:
        phi = init_weights(layers).to(device)
    opt = GD(snr=snr, approx=approx, mu_noise=mu_noise, device=device, batch_size=batch_size, seed=seed, c0=c, eps0=eps)
    if MC:
        print("MC is enabled!")
        opt.MC = MC
        opt.A = A_trans
        opt.pi = pi
        
    opt.local_clk = local_clock
    opt.spall_grad = spall_grad
        
    print(opt.snr)
    if hasattr(opt, 'nesterov_sequence'):
        opt.nesterov_sequence = True

    if hasattr(opt, 'tau'):
        if tau is not None:
            opt.tau = tau
        if p is not None:
            opt.p = p
        if q is not None:
            opt.q = q

    opt.BCD = is_BCD
    opt.p_pick = delta

    opt.start(phi)

    values_0 = [np.abs(f(opt.phi, device=device).cpu().numpy()-opt_f)]
    print("Initial Value: {} Optimal Value: {} #Params: {}".format(values_0[0], opt_f, phi.shape[1]))
    params_c = []
    params_eps = []
    if is_require_coords:
        x_0 = [opt.phi]
    if return_params:
        params_c.append(opt.gamma if hasattr(opt, 'gamma') else opt.c)
        params_eps.append(opt.eps)

    for i in tqdm(range(0, ITR_LIM, step)):
        opt.update(grad, scheduler=scheduler)
        if is_require_coords:
            x_0.append(opt.phi.cpu().numpy())
        if return_params:
            params_c.append(opt.gamma if hasattr(opt, 'gamma') else opt.c)
            params_eps.append(opt.eps)
        values_0.append(np.abs(f(opt.phi, device=device).cpu().numpy()-opt_f))


    if is_require_coords and return_params:
        return x_0, values_0, params_c, params_eps
    elif is_require_coords:
        return x_0, values_0
    elif return_params:
        return values_0, params_c, params_eps
    else:
        return values_0


def plot_var_delta(values_gd_approx, values_gd_approx_noisy, deltas, title="", snr=80, savepath=None, ylim=None):
    plt.figure(figsize=(16, 6))
    plt.suptitle(title, fontsize=25)

    plt.subplot(1, 2, 1)
    plt.title(r"SNR=$\infty$ dB", fontsize=20)
    for i in range(len(deltas)):
        plt.semilogy(values_gd_approx[i])
    plt.xlabel(r't', fontsize=20)
    plt.ylabel(r'$|J({\boldsymbol {\theta}_t}) - J({\boldsymbol {\theta}^*})|$', fontsize=20)
    plt.xticks(fontsize=14);
    plt.yticks(fontsize=14);
    if ylim is not None:
        plt.ylim(ylim)

    plt.subplot(1, 2, 2)
    plt.title(r"SNR={} dB".format(snr), fontsize=20)
    for i in range(len(deltas)):
        plt.semilogy(values_gd_approx_noisy[i], label='{}'.format(np.round(deltas[i], 2)))
    plt.xlabel(r't', fontsize=20)
    plt.ylabel(r'$|J({\boldsymbol {\theta}_t}) - J({\boldsymbol {\theta}^*})|$', fontsize=20)
    plt.xticks(fontsize=14);
    plt.yticks(fontsize=14);

    plt.legend(loc='upper right', bbox_to_anchor=(0.6, -0.15),
          fancybox=True, shadow=True, ncol=6, fontsize=14, title=r'$\rho$', title_fontsize=20)

    if savepath is None:
        return
    if ylim is not None:
        plt.ylim(ylim)
    plt.savefig(savepath, bbox_inches='tight')

    
def get_single_path(idxs, pi, N=1):
    return np.random.choice(idxs, size=N, p=pi)

def generate_markov_path(pi, A, T, num_paths=1):
    N = A.shape[0]
    idxs = np.arange(N)
    mc_paths = []
    for i in range(num_paths):
        init_idxs = get_single_path(idxs, pi)[0]
        path = [init_idxs]
        for i in range(T-1):
            path.append(get_single_path(idxs, A[path[-1]])[0])
        mc_paths.append(path)
    return mc_paths

def get_next_state(curr, A, idxs):
    nex = [get_single_path(idxs, A[i])[0] for i in curr]
    return np.unique(nex)