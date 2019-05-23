import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size,cuda=True,gain = 1):
        super(GRU, self).__init__()
        self.cuda = cuda
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.Wo = nn.Linear(hidden_size, hidden_size,bias=True)
        self.Wz = nn.Linear(input_size, hidden_size,bias=True)
        self.Ur = nn.Linear(hidden_size, hidden_size,bias=True)
        self.Wr = nn.Linear(input_size, hidden_size,bias=True)
        self.Uh = nn.Linear(hidden_size, hidden_size,bias=True)
        self.Wh = nn.Linear(input_size, hidden_size,bias=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    

        torch.nn.init.xavier_normal_(self.Wo.weight,gain=gain)
        torch.nn.init.xavier_normal_(self.Wz.weight,gain=gain)
        torch.nn.init.xavier_normal_(self.Ur.weight,gain=gain)
        torch.nn.init.xavier_normal_(self.Wr.weight,gain=gain)
        torch.nn.init.xavier_normal_(self.Uh.weight,gain=gain)
        torch.nn.init.xavier_normal_(self.Wh.weight,gain=gain)
        
    def init_hidden(self,batch_size):
        self.hidden = torch.zeros((batch_size,self.hidden_size))

        if self.cuda:
            self.hidden = self.hidden.cuda()
        
    def forward(self,x):
        self.h_last = self.hidden
        self.z = self.sigmoid(self.Wz(x) + self.Wo(self.hidden))
        self.r = self.sigmoid(self.Wr(x) + self.Ur(self.hidden))
        self.hidden = torch.mul((1-self.z),self.hidden) + torch.mul(self.z,self.tanh(self.Wh(x) + self.Uh(torch.mul(self.r,self.hidden))))
        return self.hidden,(self.z,self.r)
    
    def construct_grads(self,x,z,r):
        
        
        k = self.tanh(self.Wh(x) + self.Uh(torch.mul(r,self.h_last)))
        I = torch.eye(self.hidden_size)
        if self.cuda:
            I = I.cuda()
        self.dzt = I.mul(torch.mul(torch.mul(z,(1-z)),(k - self.h_last)))
        self.drt = I.mul(torch.mul(torch.mul(r,(1-r)),self.h_last))
        self.dht = I.mul(torch.mul(z,1-torch.mul(k,k)))
    
    def construct_PSI(self,z,r):
        I = torch.eye(self.hidden_size)
        if self.cuda:
            I = I.cuda()
        Psi = torch.matmul(self.dzt,self.Wo.weight) + I.mul(1-z) + torch.matmul(torch.matmul(self.dht,self.Uh.weight),torch.matmul(self.drt,self.Ur.weight) + I.mul(r))
        return Psi
    
    def construct_PHI(self,Ez,Er,Eh,z,r):
        ones = torch.ones(1,self.input_size)
        if self.cuda:
            ones = ones.cuda()
        b = torch.cat((ones,r),1).t()
        I = torch.eye(self.hidden_size+self.input_size)
        if self.cuda:
            I = I.cuda()
        R = I.mul(b)
        brack = torch.matmul(Eh,R) + torch.matmul(torch.matmul(self.Uh.weight,self.drt),Er)
        Phi = torch.matmul(self.dzt,Ez) + torch.matmul(self.dht,brack)
        return Phi
    
    def construct_M(self,l,i,j,x):
        l = 'z'
        x = x[:,0].unsqueeze(0)
        self.construct_grads(x,self.z,self.r)
        Psi = self.construct_PSI(self.z,self.r)
        Ez,Er,Eh = self.construct_Es(l,i,j)
        Phi = self.construct_PHI(Ez,Er,Eh,self.z,self.r)
        I = torch.eye(self.hidden_size)
        
        
        M = torch.zeros(2*self.hidden_size,2*self.hidden_size)
        if self.cuda:
            I = I.cuda()
            M = M.cuda()
        M[:self.hidden_size,:self.hidden_size] = Psi
        M[:self.hidden_size,self.hidden_size:2*self.hidden_size] = I.mul(torch.matmul(Phi,torch.cat((self.h_last,x),1).t()))
        M[self.hidden_size:2*self.hidden_size,self.hidden_size:2*self.hidden_size] = I
        return M
    def construct_Es(self,l,i,j):
        Eh = torch.zeros(self.Wh.weight.shape[0] , self.Wh.weight.shape[1]+ self.Uh.weight.shape[1])
        Er = torch.zeros(self.Wr.weight.shape[0], self.Wr.weight.shape[1]+ self.Ur.weight.shape[1])
        Ez = torch.zeros(self.Wz.weight.shape[0], self.Wz.weight.shape[1]+ self.Wo.weight.shape[1])
        if l == 'h':
            assert i < self.Wh.weight.shape[0] 
            assert j < self.Wh.weight.shape[1]+ self.Uh.weight.shape[1]
            Eh[i,j] = 1
        elif l == 'z':
            assert i < self.Wz.weight.shape[0] 
            assert j < self.Wz.weight.shape[1]+ self.Wo.weight.shape[1]
            Ez[i,j] = 1
        elif l == 'r':
            assert i < self.Wr.weight.shape[0] 
            assert j < self.Wr.weight.shape[1]+ self.Ur.weight.shape[1]
            Er[i,j] = 1
        if self.cuda:
            Ez = Ez.cuda()
            Er = Er.cuda()
            Eh = Eh.cuda()
        return Ez,Er,Eh