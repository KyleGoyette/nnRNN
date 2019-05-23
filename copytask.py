import torch
import torch.nn as nn
from optim import GeoSGD
import torch.optim as optim
import numpy as np
from LSTM import LSTM
from GRU import GRU
from RNN_Cell import OrthoRNNCell, NewOrthoRNNCell
from ComplexRNNCell import ComplexOrthoRNNCell
from exprnn import ExpRNN

import matplotlib.pyplot as plt
import os
import pickle
import argparse
from tensorboardX import SummaryWriter
import shutil
import time
import glob
import os
import copy
from geo_SGD import geoSGD
import tensorboardX
from expRNN.initialization import henaff_init,cayley_init, random_orthogonal_init
from utils import str2bool, select_network, calc_hidden_size

parser = argparse.ArgumentParser(description='auglang parameters')

parser.add_argument('--net-type', type=str, default='RNN', help='options: LSTM, RNN, DRNN, QDRNN,SRNN')
parser.add_argument('--nhid', type=int, default=128, help='hidden size of recurrent net')
parser.add_argument('--cuda', type=str2bool, default=True, help='use cuda')
parser.add_argument('--T', type=int, default=300, help='delay between sequence lengths')
parser.add_argument('--random-seed', type=int, default=400, help='random seed')
parser.add_argument('--labels', type=int, default=8, help='number of labels in the output and input')
parser.add_argument('--c-length', type=int, default=10, help='sequence length')
parser.add_argument('--alam', type=float, default=0.0001, help='alpha values lamda for ARORNN and ARORNN2')
parser.add_argument('--nonlin', type=str, default='modrelu', help='non linearity none, relu, tanh, sigmoid')
parser.add_argument('--ostep_method', type=str, default='exp', help='if learnable P, which way, exp or cayley')
parser.add_argument('--vari', type=str2bool, default=False, help='variable length')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lr_orth', type=float, default=2e-5)
parser.add_argument('--rinit', type=str, default="henaff", help='recurrent weight matrix initialization')
parser.add_argument('--iinit', type=str, default="xavier",help='input weight matrix initialization' )
parser.add_argument('--batch', type=int, default=10)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--onehot', type=str2bool, default=False)
parser.add_argument('--alpha', type=float, default=0.99)


args = parser.parse_args()

if args.rinit == "cayley":
    rinit = cayley_init
elif args.rinit == "henaff":
    rinit = henaff_init
elif args.rinit == "random":
    rinit = random_orthogonal_init
if args.iinit == "xavier":
    iinit = nn.init.xavier_normal_
elif args.iinit == 'kaiming':
    iinit = nn.init.kaiming_normal_


def generate_copying_sequence(T,labels,c_length):

    items = [1, 2, 3, 4, 5, 6, 7, 8, 0, 9]
    x = []
    y = []

    ind = np.random.randint(labels, size=c_length)
    for i in range(c_length):
        x.append([items[ind[i]]])
    for i in range(T - 1):
        x.append([items[8]])
    x.append([items[9]])
    for i in range(c_length):
        x.append([items[8]])

    for i in range(T + c_length):
        y.append([items[8]])
    for i in range(c_length):
        y.append([items[ind[i]]])

    x = np.array(x)
    y = np.array(y)

    return torch.FloatTensor([x]), torch.LongTensor([y])

def create_dataset(size, T,c_length=10):
    d_x = []
    d_y = []
    for i in range(size):
        sq_x, sq_y = generate_copying_sequence(T,8,c_length)
        sq_x, sq_y = sq_x[0], sq_y[0]
        d_x.append(sq_x)
        d_y.append(sq_y)#

    d_x = torch.stack(d_x)
    d_y = torch.stack(d_y)
    return d_x, d_y

def onehot(inp):
    #print(inp.shape)
    onehot_x = inp.new_zeros(inp.shape[0],args.labels+2)
    return onehot_x.scatter_(1,inp.long(),1)

class Model(nn.Module):
    def __init__(self,hidden_size,rec_net):
        super(Model,self).__init__()
        self.rnn = rec_net
        if isinstance(rec_net,ComplexOrthoRNNCell):
            self.lin = nn.Linear(2*hidden_size,args.labels+1)
        else:    
            self.lin = nn.Linear(hidden_size,args.labels+1)
        self.hidden_size = hidden_size
        self.params = self.rnn.params + [self.lin.weight,self.lin.bias]
        self.orthogonal_params = [self.rnn.orthogonal_params]
        self.loss_func = nn.CrossEntropyLoss()

        nn.init.xavier_normal_(self.lin.weight)

    def forward(self,x,y):
        if isinstance(self.rnn,ComplexOrthoRNNCell):
            hidden_r = hidden_i = None
        else:
            hidden = None
        outs = []
        loss = 0
        accuracy = 0
        if NET_TYPE == 'LSTM':
            self.rnn.init_states(x.shape[1])
        for i in range(len(x)):
            if isinstance(self.rnn,ComplexOrthoRNNCell):
                hidden_r, hidden_i = self.rnn.forward(x[i],hidden_r,hidden_i)
                hidden = torch.cat([hidden_r, hidden_i],dim=1)
            else:
                if args.onehot:
                    inp = onehot(x[i])
                    hidden = self.rnn.forward(inp,hidden)
                else:
                    hidden = self.rnn.forward(x[i],hidden)
            out = self.lin(hidden)
            loss += self.loss_func(out,y[i].squeeze(1))

            if i >= T + args.c_length:
                preds = torch.argmax(out,dim=1)
                actual = y[i].squeeze(1)
                correct = preds == actual
                
                accuracy += correct.sum().item()

        accuracy /= (args.c_length*x.shape[1])
        loss /= (x.shape[0])
        return loss,accuracy

def train_model(net,optimizer,batch_size,T,n_steps):

    train_losses = []
    train_accuracies = []
    save_norms = []    
    accs = []
    losses = []
    
    for i in range(n_steps):
        
        s_t = time.time()
        if args.vari:
            T = np.random.randint(1,args.T)
        x,y = create_dataset(batch_size,T,args.c_length)
        
        if CUDA:
            x = x.cuda()
            y = y.cuda()
        x = x.transpose(0,1)
        y = y.transpose(0,1)
        
        optimizer.zero_grad()
        if orthog_optimizer:
            orthog_optimizer.zero_grad()
        loss,accuracy = net.forward(x,y)
        loss_act = loss
        if NET_TYPE in ['ARORNN', 'ARORNN2'] and alam > 0:
            alpha_loss = net.rnn.alpha_loss(alam)
            loss += alpha_loss
            if writer:
                writer.add_scalar('alpha loss', alpha_loss.item(),i)

        loss.backward()
        
        norm = torch.nn.utils.clip_grad_norm_(net.params,'inf')
        save_norms.append(norm)
            

        losses.append(loss_act.item())
        if orthog_optimizer:
            net.rnn.orthogonal_step(orthog_optimizer)

        optimizer.step()
        accs.append(accuracy)
        if writer:
            writer.add_scalar('Loss', loss.item(),i)
            writer.add_scalar('Accuracy', accuracy,i)
            writer.add_scalar('Grad Norms', norm,i)

        print('Update {}, Time for Update: {} , Average Loss: {}, Accuracy: {}'.format(i +1, time.time()- s_t, loss_act.item(), accuracy))
    
    
    with open(SAVEDIR + '{}_Train_Losses'.format(NET_TYPE), 'wb') as fp:
        pickle.dump(losses,fp)

    with open(SAVEDIR + '{}_Train_Accuracy'.format(NET_TYPE),'wb') as fp:
        pickle.dump(accs,fp)

    with open(SAVEDIR + '{}_Grad_Norms'.format(NET_TYPE),'wb') as fp:
        pickle.dump(save_norms,fp)

    save_checkpoint({
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'time step': i
        },
        '{}_{}.pth.tar'.format(NET_TYPE,i)
        )
        
    return

def load_model(net, optimizer, fname):
    if fname == 'l':
        print(SAVEDIR)
        list_of_files = glob.glob(SAVEDIR + '*')
        print(list_of_files)
        latest_file = max(list_of_files, key=os.path.getctime)
        print('Loading {}'.format(latest_file))

        check = torch.load(latest_file)
        net.load_state_dict(check['state_dict'])
        optimizer.load_state_dict(check['optimizer'])
        
    else:
        check = torch.load(fname)
        net.load_state_dict(check['state_dict'])
        optimizer.load_state_dict(check['optimizer'])
    epoch = check['epoch']
    return net, optimizer,epoch

def save_checkpoint(state,fname):
    filename = SAVEDIR + fname
    torch.save(state,filename)

nonlins = ['relu','tanh','sigmoid', 'modrelu']
nonlin = args.nonlin.lower()
if nonlin not in nonlins:
    nonlin = 'none'
    print('Non lin not found, using no nonlinearity')

random_seed = args.random_seed
NET_TYPE = args.net_type
CUDA = args.cuda
alam = args.alam
decay = args.weight_decay
print(calc_hidden_size(args.net_type,23500,1,9))
#if not args.onehot:
#    hidden_size = calc_hidden_size(args.net_type,23500,1,9)
#elif args.onehot:
#    hidden_size = calc_hidden_size(args.net_type,23500,10,9)
hidden_size = args.nhid
udir = 'HS_{}_NL_{}_lr_{}_BS_{}_rinit_{}_iinit_{}_decay_{}_alpha_{}'.format(hidden_size,nonlin,args.lr,args.batch,args.rinit,args.iinit,decay,args.alpha)
if NET_TYPE in ['RORNN2', 'ARORNN2', 'NRNN2', 'NSRNN2', 'EXPRNN']:
    udir += '_lro_{}'.format(args.lr_orth)
if NET_TYPE in ['ARORNN', 'ARORNN2']:
    udir += '_aL_{}'.format(alam)
if args.onehot:
    udir = 'onehot/' + udir

if not args.vari:
    n_steps = 1500
    LOGDIR = './logs/copytask/{}/{}/{}/'.format(NET_TYPE,udir,random_seed)
    SAVEDIR = './saves/copytask/{}/{}/{}/'.format(NET_TYPE,udir,random_seed)
    print(SAVEDIR)
else:
    n_steps = 200000
    LOGDIR = './logs/varicopytask/{}/{}/{}/'.format(NET_TYPE,udir,random_seed)
    SAVEDIR = './saves/varicopytask/{}/{}/{}/'.format(NET_TYPE,udir,random_seed)
writer = None
#writer = SummaryWriter(LOGDIR)

torch.cuda.manual_seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)



inp_size = 1
T = args.T
batch_size = args.batch
out_size = args.labels + 1
if args.onehot:
    inp_size = args.labels + 2
rnn = select_network(NET_TYPE,inp_size,hidden_size,nonlin,rinit,iinit,CUDA,args.ostep_method)
net = Model(hidden_size,rnn)
if CUDA:
    net = net.cuda()
    net.rnn = net.rnn.cuda()

print('Copy task')
print(NET_TYPE)
print('Cuda: {}'.format(CUDA))
print(nonlin)
print(hidden_size)

if not os.path.exists(SAVEDIR):
    os.makedirs(SAVEDIR)

l2_norm_crit = nn.MSELoss()

orthog_optimizer = None

if args.ostep_method == 'exp' and NET_TYPE in ['ORNN2','ORNNR2','RORNN2','ARORNN2']:
    x = [
        {'params': (param for param in net.params if param is not net.rnn.log_P and param is not net.rnn.P and param is not net.rnn.UppT)},
        {'params': net.rnn.UppT, 'weight_decay': decay}
        ]
    optimizer = optim.RMSprop(x, lr=args.lr,alpha=args.alpha)
    orthog_optimizer = optim.RMSprop([net.rnn.log_P],lr=args.lr_orth,alpha=args.alpha)
elif args.ostep_method == 'cayley' and NET_TYPE in ['ORNN2','ORNNR2','RORNN2','ARORNN2']:
    optimizer = optim.RMSprop((param for param in net.params
                           if param is not net.rnn.P), lr=args.lr,alpha=args.alpha)
    orthog_optimizer = GeoSGD([net.rnn.P],lr=args.lr_orth,alpha=args.alpha)
elif args.ostep_method == 'exp' and NET_TYPE in ['NRNN2','NSRNN2']:
    x = [
        {'params': (param for param in net.params if param is not net.rnn.log_Q and param is not net.rnn.Q and param is not net.rnn.UppT)},
        {'params': net.rnn.UppT, 'weight_decay': decay}
        ]
    optimizer = optim.RMSprop(x, lr=args.lr,alpha=args.alpha)
    orthog_optimizer = optim.RMSprop([net.rnn.log_Q],lr=args.lr_orth,alpha=args.alpha)
elif args.ostep_method == 'cayley' and NET_TYPE in ['NRNN2','NSRNN2']:
    x = [
        {'params': (param for param in net.params if param is not net.rnn.log_Q and param is not net.rnn.Q and param is not net.rnn.UppT)},
        {'params': net.rnn.UppT, 'weight_decay': decay}
        ]
    optimizer = optim.RMSprop((param for param in net.params
                           if param is not net.rnn.Q), lr=args.lr,alpha=args.alpha)
    orthog_optimizer = GeoSGD([net.rnn.Q],lr=args.lr_orth,alpha=args.alpha)
elif NET_TYPE == 'EXPRNN':
    optimizer = optim.RMSprop((param for param in net.params
                           if param is not net.rnn.log_recurrent_kernel and 
                              param is not net.rnn.recurrent_kernel), lr=args.lr,alpha=args.alpha)
    orthog_optimizer = optim.RMSprop([net.rnn.log_recurrent_kernel],lr = args.lr_orth,alpha=args.alpha)
else:
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr,alpha=args.alpha)

with open(SAVEDIR + 'hparams.txt','w') as fp:
    for key,val in args.__dict__.items():
        fp.write(('{}: {}'.format(key,val)))
train_model(net,optimizer,batch_size,T,n_steps)

    
   
