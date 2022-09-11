import numpy as np
import torch
from sklearn import datasets

######################################O###################

device = "cpu"
A = torch.Tensor(np.load('data_files/A.npy')).to(device)
N = A.shape[0]

#########################################################
## DEFINE COST FUNCTION WITH IT'S DERIVATIVE

'''
## QUADRACTIC FORM

def f(x, A=A, device="cpu"):
    y = x@A@x.T
    return torch.diag(y)

def df(x, A=A, device=device):
    return (2*x@A).to(device)
'''

'''
## LOGSUMEXP

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
'''

## MLP

def get_data(batch_size=128, seed=69):
    pass

def single_layer(X, W):
    return X@W

def get_nparams(layers):
    return np.sum([prod_list(l) for l in layers])

def mlp(X, W, layers=None, bias=False, activations=None):
    mats = get_weight_matrix(W, layers)
    if bias:
        x = torch.cat([X, torch.ones((X.shape[0], 1))], dim=1)
    else:
        x = X
    for w, act, l in zip(mats, activations, layers):
        y = single_layer(x, w)
        if act is not None:
            x = act(y)
        else:
            x = y
    return x

def prod_list(l):
    i = 1
    for j in l:
        i *= j
    return i

def get_weight_matrix(W, layers):
    mats = []
    j = 0
    for l in layers:
        mats.append(torch.reshape(W[j:(j+int(prod_list(l)))], l))
    return mats

def init_weights(layers, seed=69, device="cpu"):
    torch.manual_seed(seed)

    W = torch.zeros((get_nparams(layers), 1)).to(device)
    j = 0
    for l in layers:
        w = torch.zeros(l)
        torch.nn.init.xavier_normal_(w)
        j_ = j + prod_list(l)
        W[j:j_] = torch.reshape(w, (-1, 1))
        j = j_
    return W


def f(W, batch_size=256, regression=False, device="cpu"):
    X, Y = datasets.load_breast_cancer(return_X_y=True)
    if not regression:
        nClasses = np.unique(Y).shape[0]

    if batch_size is None:
        idxs = np.arange(X.shape[0])
        np.random.shuffle(idxs)
        X = torch.Tensor(X[:batch_size, :]).to(device)
        Y = torch.Tensor(X[:batch_size, :]).to(device)
    else:
        X = torch.Tensor(X).to(device)
        Y = torch.Tensor(Y).to(device)
    M = X.shape[1]
    Y_hat = None
    loss = None
    if regression:
        Y_hat = mlp(X, W, layers=[(M+1, 1)], bias=True, activations=[None])
        loss = torch.nn.MSELoss()
    else:
        Y_hat = mlp(X, W, layers=[(M+1, nClasses)], bias=True, activations=[None])
        loss = torch.nn.CrossEntropyLoss()
    return loss(Y_hat, Y)


if __name__ == "__main__":
    layers = [(31, 2)]
    W = init_weights(layers)
    print(f(W))

