import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from RNN_Cell import OrthoRNNCell, RNNCell
from LSTM import LSTM
from expRNN.exprnn import ExpRNN
import argparse
from expRNN.initialization import (henaff_init,cayley_init,
                                   random_orthogonal_init)
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


def select_network(args, inp_size):
    iinit, rinit = get_initializers(args)
    if args.net_type == 'RNN':
        rnn = RNNCell(inp_size,args.nhid,
                      args.nonlin,
                      bias=True,
                      cuda=args.cuda,
                      r_initializer=rinit,
                      i_initializer=iinit)
    elif args.net_type == 'nnRNN':
        rnn = OrthoRNNCell(inp_size,args.nhid,args.nonlin,
                           bias=False,
                           cuda=args.cuda,
                           r_initializer=rinit,
                           i_initializer=iinit)
    elif args.net_type == 'expRNN':
        rnn = ExpRNN(inp_size,args.nhid,
                     skew_initializer=rinit,
                     input_initializer=iinit)
    elif args.net_type == 'LSTM':
        rnn = LSTM(inp_size,
                   args.nhid,
                   cuda=args.cuda)
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

def get_initializers(args):
    if args.rinit == "cayley":
        rinit = cayley_init
    elif args.rinit == "henaff":
        rinit = henaff_init
    elif args.rinit == "random":
        rinit = random_orthogonal_init
    elif args.rinit == 'xavier':
        rinit = nn.init.xavier_normal_
    if args.iinit == "xavier":
        iinit = nn.init.xavier_normal_
    elif args.iinit == 'kaiming':
        iinit = nn.init.kaiming_normal_

    return iinit, rinit

def select_optimizer(net, args):
    if args.net_type == 'nnRNN':
        x = [
            {'params': (param for param in net.parameters()
                        if param is not net.rnn.log_P
                        and param is not net.rnn.P
                        and param is not net.rnn.UppT)},
            {'params': net.rnn.UppT, 'weight_decay': args.Tdecay}
            ]
        y = [
            {'params': (param for param in net.parameters() if param is net.rnn.log_P)}
        ]
    elif args.net_type == 'expRNN':
        x = [
            {'params': (param for param in net.parameters()
                        if param is not net.rnn.log_recurrent_kernel
                        and param is not net.rnn.recurrent_kernel)}
            ]
        y = [
            {'params': (param for param in net.parameters()
                        if param is net.rnn.log_recurrent_kernel)}
        ]
    else:
        x = [
            {'params': (param for param in net.parameters())}
        ]
    if args.net_type in ['nnRNN', 'expRNN']:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(x, lr=args.lr, alpha=args.alpha)
            orthog_optimizer = optim.RMSprop(y, lr=args.lr_orth, alpha=args.alpha)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(x, lr=args.lr, alpha=args.betas)
            orthog_optimizer = optim.Adam(y, lr=args.lr_orth, betas=args.betas)
    else:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(x, lr=args.lr, alpha=args.alpha)
            orthog_optimizer = None
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(x, lr=args.lr, alpha=args.betas)
            orthog_optimizer = None
    return optimizer, orthog_optimizer


