import numpy as np
import torch

######################################O###################

device = "cpu"
A = torch.Tensor(np.load('data_files/A.npy')).to(device)
N = A.shape[0]

#########################################################
## DEFINE COST FUNCTION WITH IT'S DERIVATIVE

def f(x, A=A, device="cpu"):
    y = x@A@x.T
    return torch.diag(y)

def df(x, A=A, device=device):
    return (2*x@A).to(device)

