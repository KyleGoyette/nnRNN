import torch
import torch.nn as nn
import torch.optim as optim
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
from utils import c_mm
import math

def complex_rayleigh_init(Wr, Wi, fanin=None, gain=1):
	if not fanin:
		fanin = 1
		for p in W1.shape[1:]: fanin *= p
	scale = float(gain)/float(fanin)
	theta  = torch.empty_like(Wr).uniform_(-math.pi/2, +math.pi/2)
	rho    = np.random.rayleigh(scale, tuple(Wr.shape))
	rho    = torch.tensor(rho).to(Wr)
	Wr.data.copy_(rho*theta.cos())
	Wi.data.copy_(rho*theta.sin())


class CModReLU(torch.nn.Module):
    """ A modular ReLU activation function for complex-valued tensors """

    def __init__(self, size):
        super(CModReLU, self).__init__()
        self.bias = torch.nn.Parameter(torch.rand(1, size))
        self.relu = torch.nn.ReLU()

    def forward(self, x_re,x_im, eps=1e-5):
        """ ModReLU forward
        Args:
            x (torch.tensor): A torch float tensor with the real and imaginary part
                stored in the last dimension of the tensor; i.e. x.shape = [a, ...,b, 2]
        Kwargs:
            eps (float): A small number added to the norm of the complex tensor for
                numerical stability.
        """
        #x_re, x_im = x[..., 0], x[..., 1]
        norm = torch.sqrt(x_re ** 2 + x_im ** 2) + 1e-5
        phase_re, phase_im = x_re / norm, x_im / norm
        activated_norm = self.relu(norm + self.bias)
        return activated_norm * phase_re, activated_norm * phase_im

class ComplexLinear(torch.nn.Module):
	def __init__(self, in_features, out_features, bias=True):
		super(ComplexLinear, self).__init__()
		self.in_features  = in_features
		self.out_features = out_features
		self.Wr           = torch.nn.Parameter(torch.Tensor(out_features, in_features))
		self.Wi           = torch.nn.Parameter(torch.Tensor(out_features, in_features))
		if bias:
			self.Br = torch.nn.Parameter(torch.Tensor(out_features))
			self.Bi = torch.nn.Parameter(torch.Tensor(out_features))
		else:
			self.register_parameter('Br', None)
			self.register_parameter('Bi', None)
		self.reset_parameters()
	
	def reset_parameters(self):
		complex_rayleigh_init(self.Wr, self.Wi, self.in_features)
		if self.Br is not None and self.Bi is not None:
			self.Br.data.zero_()
			self.Bi.data.zero_()
	
	def forward(self, xr, xi):
		yrr = torch.nn.functional.linear(xr, self.Wr, self.Br)
		yri = torch.nn.functional.linear(xr, self.Wi, self.Bi)
		yir = torch.nn.functional.linear(xi, self.Wr, None)
		yii = torch.nn.functional.linear(xi, self.Wi, None)
		return yrr-yii, yri+yir


class ComplexOrthoRNNCell(nn.Module):
    def __init__(self,inp_size,hid_size,nonlin,bias=True,gain=1.0,cuda=False, ortho=True, rot = False,ref = False, alpha_rot=False,learnU=False,learn_P=False,initializer=henaff_init):
        super(ComplexOrthoRNNCell,self).__init__()
        self.cudafy = cuda
        self.hidden_size = hid_size
        self.ortho = ortho
        self.params = []
        self.orthogonal_params = []
        self.learn_P = learn_P
        self.rot = False
        #Add Non linearity
        self.m = CModReLU(hid_size)
        self.params.append(self.m.bias)
        # Create linear layer to act on input X
        self.U = ComplexLinear(inp_size,hid_size,bias=bias)
        self.params.append(self.U.Wi)
        self.params.append(self.U.Wr)
        if bias:
            self.params.append(self.U.Bi)
            self.params.append(self.U.Br)
        #nn.init.xavier_normal_(self.U.weight.data)
        #nn.init.kaiming_normal_(self.U.weight.data, nonlinearity="relu")
        # if *.1 - *.4
        
        # determine if P is learnable, if P is learnable determine how
        self.learn_P = learn_P
        if learn_P:
            assert learn_P in ['cayley','exp']
            if learn_P == 'exp':
                self.log_P_r = torch.Tensor(hid_size,hid_size)
                self.log_P_i = torch.Tensor(hid_size,hid_size)
                self.P_r = torch.Tensor(hid_size,hid_size)
                self.P_i = torch.Tensor(hid_size,hid_size)
                log_V = torch.as_tensor(initializer(hid_size))

                T,Z = linalg.schur(expm(log_V).numpy(),output='complex')
                #print(T)
                self.P_r.data = torch.from_numpy(Z.real)
                self.P_i.data = torch.from_numpy(Z.imag)
                self.log_P_r.data = torch.from_numpy(linalg.logm(Z).real)
                self.log_P_r.data = torch.from_numpy(linalg.logm(Z).imag)
                self.log_P_r = nn.Parameter(self.log_P_r)
                self.log_P_i = nn.Parameter(self.log_P_i)
                self.P_r = nn.Parameter(self.P_r)
                self.P_i = nn.Parameter(self.P_i)
                self.orthogonal_params.append(self.log_P_r)
                self.orthogonal_params.append(self.log_P_i)
                self.params.append(self.P_r)
                self.params.append(self.P_i)
                thetas = np.angle(np.diag(T))
                
                self.thetas = nn.Parameter(torch.from_numpy(thetas))
                self.register_parameter('thetas',self.thetas)
                self.params.append(self.thetas)
            elif learn_P == 'cayley':
                self.log_P = torch.as_tensor(initializer(hid_size))
                self.P = nn.Parameter(torch.Tensor(hid_size,hid_size))
                self.ortohonal_params.append(self.P)
                self.P.data = self._B(False)
                self.log_P = None

        else:
            self.log_P = torch.Tensor(hid_size,hid_size)
            self.P = torch.Tensor(hid_size,hid_size)
            self.log_P.data = torch.as_tensor(initializer(hid_size))
            
            self.P.data = self._B(False)
            
        UppT = torch.zeros(hid_size,hid_size)
        self.UppT = UppT
        #if learnU:
        #    self.UppT = nn.Parameter(self.UppT)
        #    self.params.append(self.UppT)
        self.M = torch.triu(torch.ones_like(self.UppT),diagonal=1)
        
        #for i in range(0,hid_size,):
        #    self.M[i,i+1] = 0

        # Create diagonals for *.1 and *.2
        if ref:
            z = torch.bernoulli(0.5*torch.ones(hid_size))
            x = torch.where(z==0,torch.ones_like(z),-1*torch.ones_like(z))       
            self.I = torch.diag(x)
        if not ref:
            self.I = torch.eye(hid_size)

        # Create rotations and mask M for *.3 and *.4
        self.alpha_rot = alpha_rot

        #self.thetas = [0]*int(hid_size)
        
        #self.thetas = nn.Parameter(torch.Tensor([np.random.uniform(0,2*3.14,(hid_size))]))
        #self.register_parameter('thetas',self.thetas)
        #self.params.append(self.thetas)
                

        if alpha_rot:
            self.alphas = [0]*int(hid_size)
            for i in range(0,len(self.alphas)):
                self.alphas[i] = nn.Parameter(torch.Tensor([np.random.uniform(1.00,1.00)]))
                self.register_parameter('alpha_{}'.format(i),self.alphas[i])
                self.params.append(self.alphas[i])

            

        # cudafy variables if needed
        if cuda:
            if learn_P is not False:
                self.P.data = self.P.data.cuda()
                if learn_P == 'exp':
                    self.log_P_r.data = self.log_P_r.data.cuda()
                    self.log_P_i.data = self.log_P_i.data.cuda()
            else:
                self.P_r = self.P.cuda()
            self.M = self.M.cuda()
            self.I = self.I.cuda()
            if rot:
                for item in self.thetas:
                    item = item.cuda()
            if alpha_rot:
                for item in self.alphas:
                    item = item.cuda()
    def _A(self,real,gradients=False):
        if real:
            A = self.log_P_r
        else:
            A = self.log_P_i
        if not gradients:
            A = A.data
        A = A.triu(diagonal=1)
        return A-A.t()

    def _B(self,real,gradients=False):
        return expm(self._A(real))
        
    def orthogonal_step(self,optimizer):

        if self.learn_P == 'cayley':
            optimizer.step()
            optimizer.zero_grad()
        elif self.learn_P == 'exp':
            A = self._A(True,False)
            B = self.P_r.data
            G = self.P_r.grad.data
            BtG = B.t().mm(G)
            grad = 0.5*(BtG - BtG.t())
            frechet_deriv = B.mm(expm_frechet(-A.float(), grad.float()))
            #last = torch.clone(self.log_P.data)
            self.log_P_r.grad = (frechet_deriv - frechet_deriv.t()).triu(diagonal=1).double()
            #print(self.log_P.grad)
            optimizer.step()
            #print(self.log_P.data - last)
            self.P_r.data = self._B(True,False)
            self.P_r.grad.data.zero_()

            A = self._A(False,False)
            B = self.P_i.data
            G = self.P_i.grad.data
            BtG = B.t().mm(G)
            grad = 0.5*(BtG - BtG.t())
            frechet_deriv = B.mm(expm_frechet(-A, grad))
            #last = torch.clone(self.log_P.data)
            self.log_P_i.grad = (frechet_deriv - frechet_deriv.t()).triu(diagonal=1)
            #print(self.log_P.grad)
            optimizer.step()
            #print(self.log_P.data - last)
            self.P_i.data = self._B(False,False)
            self.P_i.grad.data.zero_()

    def forward(self, x,hidden_r=None, hidden_i=None):
        if hidden_r is None:

            #print(x.shape)
            hidden_r = x.new_zeros(x.shape[0],self.hidden_size)
            hidden_i = x.new_zeros(x.shape[0],self.hidden_size)
            #print(hidden.shape)

            self.calc_I()
            if self.alpha_rot:
                self.calc_alpha_block()

        if self.alpha_rot:
            #h = self.U(x) 
            
            PUr,PUi = c_mm(self.P_r,self.P_i,torch.mul(self.UppT,self.M) + self.I_r,self.I_i)
            PuPr,PuPi = c_mm(PUr,PUi,self.P_r.t(),-self.P_i.t())
            Uxr, Uxi = self.U(x,torch.zeros_like(x))
            h_r, h_i = c_mm(hidden_r,hidden_i,PuPr.t(),-PuPi.t())
            h_r = Uxr + h_r
            h_i = Uxi + h_i
            h_r,h_i = self.m(h_r,h_i)
            return h_r, h_i
        else:
            h = self.U(x) + torch.matmul(hidden,torch.matmul(torch.matmul(self.P,torch.mul(self.UppT,self.M)+self.I),self.P.t()).t())

        return h
    
    def calc_I(self):
        self.I_r = torch.zeros_like(self.UppT)
        self.I_i = torch.zeros_like(self.UppT)
        #for i in range(0,self.hidden_size):
        #    self.I_r[i,i] = self.thetas[i].cos()
        #    self.I_i[i,i] = self.thetas[i].sin()
        self.I_r = torch.mul(torch.eye(self.hidden_size),self.thetas.cos())
        self.I_i = torch.mul(torch.eye(self.hidden_size),self.thetas.sin())
        

    def calc_alpha_block(self):
        self.alpha_block = torch.zeros_like(self.UppT)
        for i in range(0,self.hidden_size):
            self.alpha_block[i,i] = self.alphas[i]

    def row_reg_loss(self,crit,gain,lam):
        norm_loss = 0
        for row in range(self.hidden_size-1):
            norm = torch.tensor(gain)
            if self.UppT.is_cuda:
                norm = norm.cuda()
            norm_loss += lam*crit(torch.sqrt(torch.sum(torch.pow(self.UppT[row,(row+1):],2))), norm)
        return norm_loss

    def alpha_loss(self,crit,lam):
        reg_loss = 0
        for alph in range(len(self.alphas)):
            reg_loss += lam*crit(self.alphas[alph],torch.ones_like(self.alphas[alph]))
        return reg_loss