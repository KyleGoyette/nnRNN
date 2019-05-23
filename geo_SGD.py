import torch
import torch.nn as nn
import numpy as np
import time

def geoSGD(parameter, gradient, lr):
    '''
    Geodisic update using SGD involving a scaled Cayley transform
    Args:
    parameter (torch tensor) : parameter data tensor to be updated
    gradient (torch gradient tensor) : gradient associated with the parameter
    lr (float) : the learning rate
    Returns:
    update (torch tensor) : the updated values of parameter

    '''
    A = torch.matmul(gradient.t(),parameter.data) - torch.matmul(parameter.data.t(),gradient)
    I = torch.eye(parameter.shape[0])
    if A.is_cuda:
        I = I.cuda()
    cayley = torch.matmul(torch.inverse(I + (lr/2.)*A),I-(lr/2.)*A)
    update = torch.matmul(cayley,parameter.data)
    return update
