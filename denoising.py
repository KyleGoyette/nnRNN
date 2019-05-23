import torch
import torch.nn as nn
from optim import GeoSGD
import torch.optim as optim
import numpy as np
from expRNN.exprnn import ExpRNN

import pickle
import argparse
from tensorboardX import SummaryWriter
import time
import glob
from expRNN.initialization import henaff_init,cayley_init
from utils import str2bool, select_network
import os

parser = argparse.ArgumentParser(description='auglang parameters')

parser.add_argument('--net-type', type=str, default='RNN', help='options:  RNN, DRNN, QDRNN,SRNN')
parser.add_argument('--hidden-size', type=int, default=128, help='hidden size of recurrent net')
parser.add_argument('--cuda', type=str2bool, default=True, help='use cuda')
parser.add_argument('--T', type=int, default=200, help='delay between sequence lengths')
parser.add_argument('--random-seed', type=int, default=400, help='random seed')
parser.add_argument('--labels', type=int, default=9, help='number of labels in the output and input')
parser.add_argument('--c-length', type=int, default=10, help='sequence length')
parser.add_argument('--alam', type=float, default=0.0001, help='alpha values lamda for ARORNN and ARORNN2')
parser.add_argument('--nonlin', type=str, default='modrelu', help='non linearity none, relu, tanh, sigmoid')
parser.add_argument('--ostep_method', type=str, default='exp', help='if learnable P, which way, exp or cayley')
parser.add_argument('--vari', type=str2bool, default=False, help='variable length')
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--lr_orth', type=float, default=2e-4)
parser.add_argument('--rinit', type=str, default="henaff", help='recurrent weight matrix initialization')
parser.add_argument('--iinit', type=str, default="kaiming",help='input weight matrix initialization' )
parser.add_argument('--batch', type=int, default=10)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--onehot', type=str2bool, default=False)

args = parser.parse_args()
if args.rinit == "cayley":
    rinit = cayley_init
elif args.rinit == "henaff":
    rinit = henaff_init
if args.iinit == "xavier":
    iinit = nn.init.xavier_normal_
elif args.iinit == 'kaiming':
    iinit = nn.init.kaiming_normal_


def onehot(inp):
    #print(inp.shape)
    onehot_x = inp.new_zeros(inp.shape[0],args.labels+2)
    return onehot_x.scatter_(1,inp.long(),1)


def create_dataset(batch_size,T, n_sequence):
    seq = np.random.randint(1, high=args.labels, size=(batch_size, n_sequence))
    zeros1 = np.zeros((batch_size, T + n_sequence - 1))

    for i in range(batch_size):
        ind = np.random.choice(T + n_sequence - 1, n_sequence,replace=False)
        ind.sort()
        zeros1[i][ind] = seq[i]

    zeros2 = np.zeros((batch_size, T + n_sequence))
    marker = 10* np.ones((batch_size, 1))
    zeros3 = np.zeros((batch_size, n_sequence))

    x = np.concatenate((zeros1, marker, zeros3), axis=1).astype('int32')
    y = np.concatenate((zeros2, seq), axis=1).astype('int64')

    return torch.Tensor(x).unsqueeze(2), torch.LongTensor(y).unsqueeze(2)

class Model(nn.Module):
    def __init__(self,hidden_size,rec_net):
        super(Model,self).__init__()
        self.rnn = rec_net
        self.lin = nn.Linear(hidden_size,args.labels+1)
        self.hidden_size = hidden_size
        self.params = self.rnn.params + [self.lin.weight,self.lin.bias]
        self.orthogonal_params = [self.rnn.orthogonal_params]
        self.loss_func = nn.CrossEntropyLoss()

        nn.init.xavier_normal_(self.lin.weight)
        
    def forward(self,x,y):
        hidden = None
        outs = []
        loss = 0
        accuracy = 0
        for i in range(len(x)):
            if args.onehot:
                inp = onehot(x[i])
            else:
                inp = x[i]
            hidden = self.rnn.forward(inp,hidden)
            out = self.lin(hidden)
            loss += self.loss_func(out,y[i].squeeze(1))

            if i > T + args.c_length:
                
                preds = torch.argmax(out,dim=1)
                actual = y[i].squeeze(1)

                correct = preds == actual
                
                accuracy += correct.sum().item()
        accuracy /= (args.c_length*x.shape[1])
        loss /= (x.shape[0])
        return loss,accuracy

    def loss(self, logits, y):
        print(logits.shape)
        print(y.shape)
        print(logits.view(-1,9))
        return self.loss_func(logits.view(-1, 9), y.view(-1))
    
    def accuracy(self, logits, y):
        preds = torch.argmax(logits,dim=2)[:,T+args.c_length:]

        return torch.eq(preds, y[:,T+args.c_length:]).float().mean()

def train_model(net,optimizer,batch_size,T):
    train_losses = []
    train_accuracies = []
    save_norms = []    
    accs = []
    losses = []
    
    for i in range(200000):
        
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

        if NET_TYPE in ['ARORNN', 'ARORNN2'] and alam > 0:
            alpha_loss = net.rnn.alpha_loss(alam)
            loss += alpha_loss


        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.params,'inf')
        save_norms.append(norm)
        writer.add_scalar('Grad Norms', norm,i)

        losses.append(loss.item())
        if orthog_optimizer:
            net.rnn.orthogonal_step(orthog_optimizer)

        optimizer.step()
        accs.append(accuracy)
        writer.add_scalar('Loss', loss.item(),i)
        writer.add_scalar('Accuracy', accuracy,i)

        print('Update {}, Time for Update: {} , Average Loss: {}, Accuracy: {}'.format(i +1, time.time()- s_t, loss.item(), accuracy))
    
    
    with open(SAVEDIR + '{}_Train_Losses'.format(NET_TYPE), 'wb') as fp:
        pickle.dump(losses,fp)

    with open(SAVEDIR + '{}_Train_Accuracy'.format(NET_TYPE),'wb') as fp:
        pickle.dump(accs,fp)

    with open(SAVEDIR + '{}_Grad_Norms'.format(NET_TYPE),'wb') as fp:
        pickle.dump(save_norms,fp)

    save_checkpoint({
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_step': i
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
udir = 'HS_{}_NL_{}_lr_{}_BS_{}_rinit_{}_iinit_{}_decay_{}'.format(args.hidden_size,nonlin,args.lr,args.batch,args.rinit,args.iinit,decay)
if NET_TYPE in ['RORNN2', 'ARORNN2', 'NRNN2', 'NSRNN2']:
    udir += '_lro_{}'.format(args.lr_orth)
if NET_TYPE in ['ARORNN', 'ARORNN2']:
    udir += '_aL_{}'.format(alam)

if args.onehot:
    udir = 'onehot/' + udir
LOGDIR = './logs/denoisetask/{}/{}/{}/'.format(NET_TYPE,udir,random_seed)
SAVEDIR = './saves/denoisetask/{}/{}/{}/'.format(NET_TYPE,udir,random_seed)
if not os.path.exists(SAVEDIR):
    os.makedirs(SAVEDIR)
with open(SAVEDIR + 'hparams.txt','w') as fp:
    for key,val in args.__dict__.items():
        fp.write(('{}: {}'.format(key,val)))

writer = SummaryWriter(LOGDIR)

torch.cuda.manual_seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)


if args.onehot:
    inp_size = args.labels + 2
else:
    inp_size = 1
hid_size = args.hidden_size
T = args.T
batch_size = args.batch
out_size = args.labels + 1

rnn = select_network(NET_TYPE,inp_size,hid_size,nonlin,rinit,iinit,CUDA,args.ostep_method)

net = Model(hid_size,rnn)
if CUDA:
    net = net.cuda()
    net.rnn = net.rnn.cuda()

print('Denoise task')
print(NET_TYPE)
print('Cuda: {}'.format(CUDA))
print(nonlin)



l2_norm_crit = nn.MSELoss()

orthog_optimizer = None

if args.ostep_method == 'exp' and NET_TYPE in ['ORNN2','ORNNR2','RORNN2','ARORNN2']:
    x = [
        {'params': (param for param in net.params if param is not net.rnn.log_P and param is not net.rnn.P and param is not net.rnn.UppT)},
        {'params': net.rnn.UppT, 'weight_decay': decay}
        ]
    optimizer = optim.RMSprop(x, lr=args.lr)
    orthog_optimizer = optim.RMSprop([net.rnn.log_P],lr=args.lr_orth)
elif args.ostep_method == 'cayley' and NET_TYPE in ['ORNN2','ORNNR2','RORNN2','ARORNN2']:
    optimizer = optim.RMSprop((param for param in net.params
                           if param is not net.rnn.P), lr=args.lr)
    orthog_optimizer = GeoSGD([net.rnn.P],lr=args.lr_orth)
elif args.ostep_method == 'exp' and NET_TYPE in ['NRNN2','NSRNN2']:
    x = [
        {'params': (param for param in net.params if param is not net.rnn.log_Q and param is not net.rnn.Q and param is not net.rnn.UppT)},
        {'params': net.rnn.UppT, 'weight_decay': decay}
        ]
    optimizer = optim.RMSprop(x, lr=args.lr)
    orthog_optimizer = optim.RMSprop([net.rnn.log_Q],lr=args.lr_orth)
elif args.ostep_method == 'cayley' and NET_TYPE in ['NRNN2','NSRNN2']:
    x = [
        {'params': (param for param in net.params if param is not net.rnn.log_Q and param is not net.rnn.Q and param is not net.rnn.UppT)},
        {'params': net.rnn.UppT, 'weight_decay': decay}
        ]
    optimizer = optim.RMSprop((param for param in net.params
                           if param is not net.rnn.Q), lr=args.lr)
    orthog_optimizer = GeoSGD([net.rnn.Q],lr=args.lr_orth)
elif NET_TYPE == 'EXPRNN':
    optimizer = optim.RMSprop((param for param in net.params
                           if param is not net.rnn.log_recurrent_kernel and 
                              param is not net.rnn.recurrent_kernel), lr=args.lr)
    orthog_optimizer = optim.RMSprop([net.rnn.log_recurrent_kernel],lr = args.lr_orth)
else:
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr)
train_model(net,optimizer,batch_size,T)

    
   
