import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Function,gradcheck, Variable
import torch.functional as F
from scipy import linalg
import numpy as np
import time
import copy
import pickle
from DataSaver import DataSaver
from expRNN.exprnn import modrelu
from expRNN.initialization import henaff_init, cayley_init
from expRNN.exp_numpy import expm, expm_frechet
import math
verbose = False

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


class RNNCell(nn.Module):
    def __init__(self,inp_size,hid_size,nonlin,bias=True,cuda=False,r_initializer=henaff_init,i_initializer=nn.init.xavier_normal_):
        super(RNNCell,self).__init__()
        self.cudafy = cuda
        self.hidden_size = hid_size        
        self.params = []
        self.orthogonal_params = []
        #Add Non linearity
        if nonlin == 'relu':
            self.nonlinearity = nn.ReLU()
        if nonlin == 'modrelu':
            self.nonlinearity = modrelu(hid_size)
            self.params.append(self.nonlinearity.b)
        elif nonlin == 'tanh':
            self.nonlinearity = nn.Tanh()
        elif nonlin == 'sigmoid':
            self.nonlinearity = nn.Sigmoid()
        else:
            self.nonlinearity = None

        # Create linear layer to act on input X
        self.U = nn.Linear(inp_size,hid_size,bias=bias)
        self.params.append(self.U.weight)
        if bias:
            self.params.append(self.U.bias)
        self.i_initializer = i_initializer

        self.V = nn.Linear(hid_size,hid_size)
        nn.init.xavier_normal_(self.V.weight.data)
        self.V.weight.data = torch.as_tensor(r_initializer(hid_size),
                        dtype=self.V.weight.data.dtype)
        A = self.V.weight.data.triu(diagonal=1)
        A = A - A.t()
        self.V.weight.data = expm(A)
        self.ortho = False
        self.params.append(self.V.weight)
        self.params.append(self.V.bias)

        self.r_initializer = r_initializer
        self.reset_parameters()
        

    def reset_parameters(self):
        self.i_initializer(self.U.weight.data)
        self.V.data = torch.as_tensor(self.r_initializer(self.hidden_size))

    def forward(self,x,hidden = None):
        if hidden is None:
            hidden = x.new_zeros(x.shape[0],self.hidden_size)

        h = self.U(x) + self.V(hidden)
        if self.nonlinearity:
            h = self.nonlinearity(h)
        return h
        



class OrthoRNNCell(nn.Module):
    def __init__(self,inp_size,hid_size,nonlin,bias=True,cuda=False, ortho=True, rot = False,ref = False, alpha_rot=False,learnU=True,ostep_method=False,r_initializer=henaff_init,i_initializer=nn.init.xavier_normal_):
        super(OrthoRNNCell,self).__init__()
        self.cudafy = cuda
        self.hidden_size = hid_size
        self.ortho = ortho
        self.params = []
        self.orthogonal_params = []
        self.ostep_method = ostep_method
        #Add Non linearity
        if nonlin == 'relu':
            self.nonlinearity = nn.ReLU()
        if nonlin == 'modrelu':
            self.nonlinearity = modrelu(hid_size)
            self.params.append(self.nonlinearity.b)
        elif nonlin == 'tanh':
            self.nonlinearity = nn.Tanh()
        elif nonlin == 'sigmoid':
            self.nonlinearity = nn.Sigmoid()
        else:
            self.nonlinearity = None

        # Create linear layer to act on input X
        self.U = nn.Linear(inp_size,hid_size,bias=bias)
        self.params.append(self.U.weight)
        if bias:
            self.params.append(self.U.bias)

        self.i_initializer = i_initializer
        self.r_initializer = r_initializer
        self.row_reg_crit = nn.MSELoss()
        
        # determine if P is learnable, if P is learnable determine how
        self.ostep_method = ostep_method
        if ostep_method is not False:
            assert ostep_method in ['cayley','exp']
            if ostep_method == 'exp':
                self.log_P = torch.Tensor(hid_size,hid_size)
                self.log_P = nn.Parameter(self.log_P)
                self.orthogonal_params.append(self.log_P)

                self.P = torch.Tensor(hid_size,hid_size)
                self.P = nn.Parameter(self.P)
                
            elif ostep_method == 'cayley':
                self.P = nn.Parameter(torch.Tensor(hid_size,hid_size))
                self.orthogonal_params.append(self.P)

        else:
            self.P = torch.Tensor(hid_size,hid_size)
            
        
        UppT = torch.zeros(hid_size,hid_size)
        self.UppT = UppT
        if learnU:
            self.UppT = nn.Parameter(self.UppT)
            self.params.append(self.UppT)
        self.M = torch.triu(torch.ones_like(self.UppT),diagonal=1)
        # Create diagonals for *.1 and *.2
        if ref:
            z = torch.bernoulli(0.5*torch.ones(hid_size))
            x = torch.where(z==0,torch.ones_like(z),-1*torch.ones_like(z))       
            self.D = torch.diag(x)
        if not ref:
            self.D = torch.eye(hid_size)

        # Create rotations and mask M for *.3 and *.4
        self.rot = rot
        self.alpha_rot = alpha_rot
        if rot:
            assert not ref
            self.thetas = [0]*int(hid_size)
            for i in range(0,len(self.thetas)):
                self.thetas[i] = nn.Parameter(torch.Tensor([np.random.uniform(0,2*3.14)]))
                self.register_parameter('theta_{}'.format(i),self.thetas[i])
                self.params.append(self.thetas[i])
                

            if alpha_rot:
                self.alpha_crit = nn.MSELoss()
                self.alphas = [0]*int(hid_size/2)
                for i in range(0,len(self.alphas)):
                    self.alphas[i] = nn.Parameter(torch.Tensor([np.random.uniform(1.00,1.00)]))
                    self.register_parameter('alpha_{}'.format(i),self.alphas[i])
                    self.params.append(self.alphas[i])

        
        self.reset_parameters()

        # cudafy variables if needed
        if cuda:
            if ostep_method is not False:
                self.P.data = self.P.data.cuda()
                if ostep_method == 'exp':
                    self.log_P.data = self.log_P.data.cuda()
            else:
                self.P = self.P.cuda()
            self.M = self.M.cuda()
            self.D = self.D.cuda()
            if rot:
                for item in self.thetas:
                    item = item.cuda()
            if alpha_rot:
                for item in self.alphas:
                    item = item.cuda()

    def reset_parameters(self):
        self.i_initializer(self.U.weight.data)
        if self.ostep_method == 'exp':
            self.log_P.data = torch.as_tensor(self.r_initializer(self.hidden_size))
            self.P.data = self._B(False)
        elif self.ostep_method == 'cayley' or not self.ostep_method:
            self.log_P = torch.as_tensor(self.r_initializer(self.hidden_size))
            self.P.data = self._B(False)
            self.log_P = None

    def _A(self,gradients=False):
        A = self.log_P
        if not gradients:
            A = A.data
        A = A.triu(diagonal=1)
        return A-A.t()

    def _B(self,gradients=False):
        return expm(self._A())
        
    def orthogonal_step(self,optimizer):

        if self.ostep_method == 'cayley':
            optimizer.step()
            optimizer.zero_grad()
        elif self.ostep_method == 'exp':
            A = self._A(False)
            B = self.P.data
            G = self.P.grad.data
            BtG = B.t().mm(G)
            grad = 0.5*(BtG - BtG.t())
            frechet_deriv = B.mm(expm_frechet(-A, grad))
            #last = torch.clone(self.log_P.data)
            self.log_P.grad = (frechet_deriv - frechet_deriv.t()).triu(diagonal=1)
            #print(self.log_P.grad)
            optimizer.step()
            #print(self.log_P.data - last)
            self.P.data = self._B(False)
            self.P.grad.data.zero_()

    def forward(self, x,hidden=None):
        if hidden is None:
            hidden = x.new_zeros(x.shape[0],self.hidden_size)
            self.calc_rec()
            
                

        if self.alpha_rot:
            h = self.U(x) + torch.matmul(hidden,self.rec)
        else:
            h = self.U(x) + torch.matmul(hidden,self.rec)
        

        if self.nonlinearity:
                h = self.nonlinearity(h)

        return h

    def calc_rec(self):
        if self.rot:
            self.calc_D()
            if self.alpha_rot:
                self.calc_alpha_block()
                self.rec = torch.matmul(torch.matmul(self.P,torch.mul(self.UppT,self.M)+torch.mul(self.alpha_block,self.D)),self.P.t())
            else:
                self.rec = torch.matmul(torch.matmul(self.P,torch.mul(self.UppT,self.M)+self.D),self.P.t())
    
    def calc_D(self):
        self.D = torch.zeros_like(self.UppT)
        for i in range(0,self.hidden_size,2):
            self.D[i,i] = self.thetas[int(i/2)].cos()
            self.D[i,i+1] = -self.thetas[int(i/2)].sin()
            self.D[i+1,i] = self.thetas[int(i/2)].sin()
            self.D[i+1,i+1] = self.thetas[int(i/2)].cos()


    def calc_alpha_block(self):
        self.alpha_block = torch.zeros_like(self.UppT)
        for i in range(0,self.hidden_size,2):
            self.alpha_block[i,i] = self.alphas[int(i/2)]
            self.alpha_block[i+1,i] = self.alphas[int(i/2)]
            self.alpha_block[i,i+1] = self.alphas[int(i/2)]
            self.alpha_block[i+1,i+1] = self.alphas[int(i/2)]

    def row_reg_loss(self,gain,lam):
        norm_loss = 0
        for row in range(self.hidden_size-1):
            norm = torch.tensor([gain])
            if self.UppT.is_cuda:
                norm = norm.cuda()
            norm_loss += lam*self.row_reg_crit(torch.sqrt(torch.sum(torch.pow(self.UppT[row,(row+1):],2))), norm)
        return norm_loss

    def alpha_loss(self,lam):
        reg_loss = 0
        for alph in range(len(self.alphas)):
            reg_loss += lam*self.alpha_crit(self.alphas[alph],torch.ones_like(self.alphas[alph]))
        return reg_loss


class NewOrthoRNNCell(nn.Module):
    def __init__(self,inp_size,hid_size,nonlin,bias=True,cuda=False,learnU=True,orth_method = 'exp',r_initializer=henaff_init,i_initializer=nn.init.xavier_normal_,schur=False):
        super(NewOrthoRNNCell,self).__init__()
        self.cudafy = cuda
        self.hidden_size = hid_size
        self.params = []
        self.orthogonal_params = []
        self.orth_method = orth_method
        self.schur = schur
        #Add Non linearity
        if nonlin == 'relu':
            self.nonlinearity = nn.ReLU()
        if nonlin == 'modrelu':
            self.nonlinearity = modrelu(hid_size)
            self.params.append(self.nonlinearity.b)
        elif nonlin == 'tanh':
            self.nonlinearity = nn.Tanh()
        elif nonlin == 'sigmoid':
            self.nonlinearity = nn.Sigmoid()
        else:
            self.nonlinearity = None

        # Create linear layer to act on input X
        self.U = nn.Linear(inp_size,hid_size,bias=bias)
        self.params.append(self.U.weight)
        if bias:
            self.params.append(self.U.bias)

        self.i_initializer =  i_initializer
        self.r_initializer = r_initializer
        
        assert orth_method in ['cayley','exp']
        if orth_method == 'exp':
            self.log_Q = torch.Tensor(hid_size,hid_size)
            self.Q = torch.Tensor(hid_size,hid_size)
            self.log_Q = nn.Parameter(self.log_Q)
            self.Q = nn.Parameter(self.Q)
            self.orthogonal_params.append(self.log_Q)
            #self.params.append(self.Q)
        elif orth_method == 'cayley':
            self.log_Q = torch.as_tensor(r_initializer(hid_size))
            self.Q = nn.Parameter(torch.Tensor(hid_size,hid_size))
            self.ortohonal_params.append(self.P)
            self.P.data = self._B(False)
            self.log_P = None
            
        UppT = torch.zeros(hid_size,hid_size)
        self.UppT = UppT
        if learnU:
            self.UppT = nn.Parameter(self.UppT)
            self.params.append(self.UppT)
        self.M = torch.triu(torch.ones_like(self.UppT),diagonal=1)

        self.reset_parameters()

        if cuda:
            self.M = self.M.cuda()
            self.UppT.data = self.UppT.data.cuda()
            self.U = self.U.cuda()
            self.Q.data = self.Q.data.cuda()
            if orth_method == 'exp':
                self.log_Q.data = self.log_Q.data.cuda()
            else:
                self.Q = self.Q.cuda()                

        self.row_reg_crit = nn.MSELoss()


    def reset_parameters(self):
        self.i_initializer(self.U.weight.data)
        self.log_Q.data = torch.as_tensor(self.r_initializer(self.hidden_size))
        self.Q.data = self._B(False)

    def _A(self,gradients=False):
        A = self.log_Q
        if not gradients:
            A = A.data
        A = A.triu(diagonal=1)
        return A-A.t()

    def _B(self,gradients=False):
        return expm(self._A())
        
    def orthogonal_step(self,optimizer):

        if self.orth_method == 'cayley':
            optimizer.step()
            optimizer.zero_grad()
        elif self.orth_method == 'exp':
            A = self._A(False)
            B = self.Q.data
            G = self.Q.grad.data
            BtG = B.t().mm(G)
            grad = 0.5*(BtG - BtG.t())
            frechet_deriv = B.mm(expm_frechet(-A, grad))
            self.log_Q.grad = (frechet_deriv - frechet_deriv.t()).triu(diagonal=1)
            optimizer.step()
            self.Q.data = self._B(False)
            self.Q.grad.data.zero_()
            optimizer.zero_grad()

    def calc_P(self):
        T,P = linalg.schur(self.Q.detach().cpu().numpy())
        self.P = torch.Tensor(P)
        if self.cuda:
            self.P = self.P.cuda()

    def forward(self, x,hidden=None):
        if hidden is None:
            hidden = x.new_zeros(x.shape[0],self.hidden_size)
            if self.schur:
                self.calc_P()

            
        if self.schur:
            h = self.U(x) + torch.matmul(hidden,self.Q + torch.matmul(torch.matmul(self.P,torch.mul(self.UppT,self.M)),self.P.t()))
        else:    
            h = self.U(x) + torch.matmul(hidden,self.Q + torch.matmul(torch.matmul(self.Q,torch.mul(self.UppT,self.M)),self.Q.t()))
        if self.nonlinearity:
            h = self.nonlinearity(h)
        return h

    def row_reg_loss(self,gain,lam):
        norm_loss = 0
        for row in range(self.hidden_size-1):
            norm = torch.Tensor([gain])
            if self.UppT.is_cuda:
                norm = norm.cuda()
            norm_loss += lam*self.row_reg_crit(torch.sqrt(torch.sum(torch.pow(self.UppT[row,(row+1):],2))), norm)
        return norm_loss