import numpy as np
import scipy.linalg as linalg
import torch.nn as nn
import torch
from RNN_Cell import OrthoRNNCell, NewOrthoRNNCell,RNNCell
from LSTM import LSTM
from expRNN.exprnn import ExpRNN
def rvs(dim=3):
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim-n+1,))
        D[n-1] = np.sign(x[0])
        x[0] -= D[n-1]*np.sqrt((x*x).sum())
        # Householder transformation
        Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
        mat = np.eye(dim)
        mat[n-1:, n-1:] = Hx
        H = np.dot(H, mat)
    # Fix the last sign such that the determinant is 1
    D[-1] = (-1)**(1-(dim % 2))*D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D*H.T).T
    return H

def c_mm(A_r,A_i,B_r,B_i):
    C_r = torch.mm(A_r,B_r) - torch.mm(A_i,B_i)
    C_i = torch.mm(A_i,B_r) + torch.mm(A_r,B_i)
    return C_r,C_i
def star(A_r,A_i):
    return A_r.t(),-A_i.t()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def select_network(net_type,inp_size,hid_size,nonlin,rinit,iinit,cuda,ostep_method):
    if net_type == 'RNN':
        rnn = RNNCell(inp_size,hid_size,nonlin,bias=False,cuda=cuda,r_initializer=rinit,i_initializer=iinit)
    elif net_type == 'ORNN':
        rnn = OrthoRNNCell(inp_size,hid_size,nonlin,bias=False, cuda=cuda,ostep_method=False)
    elif net_type == 'ORNNR':
        rnn = OrthoRNNCell(inp_size,hid_size,nonlin,bias=False,cuda=cuda,ref=True,ostep_method=False)
    elif net_type == 'RORNN':
        rnn = OrthoRNNCell(inp_size,hid_size,nonlin,bias=False, cuda=cuda,rot=True,ostep_method=False)
    elif net_type == 'ARORNN':
        rnn = OrthoRNNCell(inp_size,hid_size,nonlin,bias=False, cuda=cuda,rot=True,alpha_rot=True,ostep_method=False)
    elif net_type == 'ORNN2':
        rnn = OrthoRNNCell(inp_size,hid_size,nonlin,bias=False,cuda=cuda,ostep_method=ostep_method,r_initializer=rinit,i_initializer=iinit)
    elif net_type == 'ORNNR2':
        rnn = OrthoRNNCell(inp_size,hid_size,nonlin,bias=False,cuda=cuda,ref=True,ostep_method=ostep_method,r_initializer=rinit,i_initializer=iinit)
    elif net_type == 'RORNN2':
        rnn = OrthoRNNCell(inp_size,hid_size,nonlin,bias=False,cuda=cuda,rot=True,ostep_method=ostep_method,r_initializer=rinit,i_initializer=iinit)
    elif net_type == 'ARORNN2':
        rnn = OrthoRNNCell(inp_size,hid_size,nonlin,bias=False,cuda=cuda,rot=True,alpha_rot=True,ostep_method=ostep_method,r_initializer=rinit,i_initializer=iinit)
    elif net_type == 'NRNN2':
        rnn = NewOrthoRNNCell(inp_size,hid_size,nonlin,bias=False,cuda=cuda,orth_method=ostep_method,r_initializer=rinit,i_initializer=iinit)
    elif net_type == 'NSRNN2':
        rnn = NewOrthoRNNCell(inp_size,hid_size,nonlin,bias=False,cuda=cuda,orth_method=ostep_method,r_initializer=rinit,i_initializer=iinit,schur=True)
    elif net_type == 'EXPRNN':
        rnn = ExpRNN(inp_size,hid_size,skew_initializer=rinit,input_initializer=iinit)
    elif net_type == 'LSTM':
        rnn = LSTM(inp_size,hid_size,cuda=cuda)
    return rnn

def calc_hidden_size(net_type, n_params, n_in, n_out):

    if net_type == 'RNN':
        a = 1
        b = n_in + n_out
        c = - n_params + n_out 

    elif net_type in ['RORNN2', 'ARORNN2','NRNN2', 'NSRNN2']:
        a = 1
        b = n_in + n_out - 1/2
        c = -n_params + n_out

    elif net_type in ['EXPRNN']:
        a = 0.5
        b = n_in + n_out
        c = -n_params + n_out
    elif net_type == 'LSTM':
        a = 4
        b = 4*(n_in + n_out)
        c = -n_params + n_out
    
    return int(np.roots([a,b,c])[1])


def calc_hidden_size_PTB(net_type, n_params, n_chars, n_emb):

    if net_type == 'RNN':
        a = 1
        b = n_chars + n_emb 
        c = - n_params + n_chars + n_chars*n_emb 

    elif net_type in ['RORNN2', 'ARORNN2','NRNN2', 'NSRNN2']:
        a = 1
        b = n_emb + n_chars - 1/2
        c = - n_params + n_chars + n_chars*n_emb 

    elif net_type in ['EXPRNN']:
        a = 0.5
        b = n_emb + n_chars
        c = - n_params + n_chars + n_chars*n_emb 
    elif net_type == 'LSTM':
        a = 4
        b = 4*(n_emb + n_chars)
        c = -n_params + n_chars + n_chars*n_emb 
    
    return int(np.roots([a,b,c])[1])


def retrieve_weight_matrices(path,test):
    data = torch.load(path)
    