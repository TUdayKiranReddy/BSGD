import numpy as np
import torch

######################################O###################

device = "cpu"
A = torch.Tensor(np.load('data_files/A.npy')).to(device)
N = A.shape[0]

#########################################################
## DEFINE COST FUNCTION WITH IT'S DERIVATIVE
'''
def f(x, A=A, device="cpu"):
    y = x@A@x.T
    return torch.diag(y)

def df(x, A=A, device=device):
    return (2*x@A).to(device)
'''

def log_sum_exp(x, temp=1.0):
    m = torch.max(x, dim=1)[0].unsqueeze(1)
    Y = temp*(x - m.repeat(1, x.shape[1]))
    return torch.log(torch.sum(torch.exp(Y), axis=1)) + torch.log(m)

def softmax(x, temp=1.0):
    m = torch.max(x, dim=1)[0].unsqueeze(1)
    Y = temp*(x - m.repeat(1, x.shape[1]))
    S = torch.sum(torch.exp(Y), axis=1)
    return torch.exp(Y)/S.unsqueeze(1)

def d_log_sum_exp(x, temp=1.0):
    return temp*softmax(x, temp=temp)
    
def f(x, A=A, temp=1.0, device=device):
    y = x@A@x.T + torch.logsumexp(temp*x, dim=1)
    return torch.diag(y) - 6.9052555215155556

def df(x, A=A, temp=1.0, device=device):
    return (2*x@A + temp*torch.nn.functional.softmax(x, dim=1)).to(device)
