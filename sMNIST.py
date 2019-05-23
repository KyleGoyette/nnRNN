import torch
import torch.nn as nn
from optim import GeoSGD
import torch.optim as optim
import torchvision as T
import numpy as np
from LSTM import LSTM
from GRU import GRU
from RNN_Cell import OrthoRNNCell, NewOrthoRNNCell
from tensorboardX import SummaryWriter
import os
import pickle
import argparse
import shutil
import time
import glob
import os
import copy
from exprnn import ExpRNN
from expRNN.initialization import henaff_init,cayley_init, random_orthogonal_init
from utils import str2bool, select_network, calc_hidden_size
from torch._utils import _accumulate
from torch.utils.data import Subset

parser = argparse.ArgumentParser(description='auglang parameters')

parser.add_argument('--net-type', type=str, default='RNN', help='options: RNN, ORNN, ORNNR, RORNN, ARORNN')
parser.add_argument('--nhid', type=int, default=400, help='hidden size of recurrent net')
parser.add_argument('--save-freq', type=int, default=50, help='frequency to save data')
parser.add_argument('--cuda', type=str2bool, default=True, help='use cuda')
parser.add_argument('--random-seed', type=int, default=400, help='random seed')
parser.add_argument('--permute', type=str2bool, default=True, help='permute the order of sMNIST')
parser.add_argument('--alam', type=float, default=0.0001, help='alpha values lamda for ARORNN and ARORNN2')
parser.add_argument('--nonlin', type=str, default='modrelu', help='non linearity none, relu, tanh, sigmoid')
parser.add_argument('--ostep_method', type=str, default='exp', help='if learnable P, which way, exp or cayley')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lr_orth', type=float, default=2e-5)
parser.add_argument('--rinit', type=str, default="henaff", help='recurrent weight matrix initialization')
parser.add_argument('--iinit', type=str, default="kaiming",help='input weight matrix initialization' )
parser.add_argument('--batch', type=int, default=100)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--alpha', type=float, default=0.99)

args = parser.parse_args()

torch.cuda.manual_seed(args.random_seed)
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

if args.permute:
    rng = np.random.RandomState(1234)
    order = rng.permutation(784)
else:
    order = np.arange(784)


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

trainset = T.datasets.MNIST(root='./MNIST', train=True, download=True, transform=T.transforms.ToTensor())
valset = T.datasets.MNIST(root='./MNIST', train=True, download=True, transform=T.transforms.ToTensor())
offset = 10000
R = rng.permutation(len(trainset))
lengths = (len(trainset) - offset, offset)
trainset,valset = [Subset(trainset, R[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]
testset = T.datasets.MNIST(root='./MNIST', train=False, download=True, transform=T.transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=False, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch, num_workers=2)


class Model(nn.Module):
    def __init__(self,hidden_size,rnn):
        super(Model, self).__init__()
        self.rnn = rnn
        self.hidden_size = hidden_size
        self.lin = nn.Linear(hidden_size,10)
        self.loss_func = nn.CrossEntropyLoss()
        self.params = rnn.params + [self.lin.weight,self.lin.bias]

    def forward(self,inputs,y,order):
        h = None
        #print(inputs.shape)
        inputs = inputs[:,order]
        for input in torch.unbind(inputs,dim=1):
            h = self.rnn(input.unsqueeze(1),h)
        out = self.lin(h)
        
        loss = self.loss_func(out,y)
        preds = torch.argmax(out,dim=1)
        correct = torch.eq(preds,y).sum().item()
        return loss, correct

def test_model(net,dataloader):
    accuracy = 0
    loss = 0
    net.eval()
    with torch.no_grad():   
        for i, data in enumerate(dataloader):
            
            x,y = data
            x = x.view(-1,784)
            if CUDA:
                x = x.cuda()
                y = y.cuda()
            if NET_TYPE == 'LSTM':
                net.rnn.init_states(x.shape[0])
            
            loss,c = net.forward(x,y,order)

            accuracy += c
        
    accuracy /= len(testset)
    return loss,accuracy

def save_checkpoint(state,fname):
    filename = os.path.join(SAVEDIR,fname)
    torch.save(state,filename)

def train_model(net,optimizer,num_epochs):

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    save_norms = []
    best_test_acc = 0
    for epoch in range(0,num_epochs):
        s_t = time.time()
        accs = []
        losses = []
        norms = []
        norm_losses = []
        processed = 0
        alpha_losses = []
        net.train()
        correct = 0
        for i,data in enumerate(trainloader,0):
            inp_x, inp_y = data
            inp_x = inp_x.view(-1,784)

            if CUDA:
                inp_x = inp_x.cuda()
                inp_y = inp_y.cuda()
            
            if NET_TYPE == 'LSTM':
                net.rnn.init_states(inp_x.shape[0])

            optimizer.zero_grad()
            if orthog_optimizer:
                orthog_optimizer.zero_grad()
            
            loss, c = net.forward(inp_x,inp_y,order)
            correct += c
            processed += inp_x.shape[0]

            accs.append(correct/float(processed))
            
            #calculate losses for orthogonal rnn and alpha blocks
            if NET_TYPE in ['ARORNN', 'ARORNN2'] and alam > 0:
                alpha_loss = net.rnn.alpha_loss(alam)
                loss += alpha_loss
                alpha_losses.append(alpha_loss.item())

            loss.backward()
            losses.append(loss.item())
            
            if orthog_optimizer:
                net.rnn.orthogonal_step(orthog_optimizer)

            optimizer.step()

            norm = torch.nn.utils.clip_grad_norm_(net.parameters(),'inf')
            norms.append(norm)
        
        test_loss, test_acc = test_model(net,valloader)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        if test_acc > best_test_acc:
            save_checkpoint({
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
                },
                '{}.pth.tar'.format('best_model')
                )
   

        print('Epoch {}, Time for Epoch: {}, Train Loss: {}, Train Accuracy: {} Test Loss: {} Test Accuracy {}'.format(epoch +1, time.time()- s_t, np.mean(losses),  np.mean(accs), test_loss, test_acc))
        train_losses.append(np.mean(losses))
        train_accuracies.append(np.mean(accs))
        save_norms.append(np.mean(norms))
        
        # write data to tensorboard
        writer.add_scalar('Train Loss', train_losses[-1],epoch)
        writer.add_scalar('Train Accuracy', train_accuracies[-1],epoch)
        writer.add_scalar('Test Loss', test_losses[-1],epoch)
        writer.add_scalar('Test Accuracy', test_accuracies[-1],epoch)
        writer.add_scalar('Gradient Norms', save_norms[-1],epoch)
        #if net.rnn.ortho:
        #    writer.add_scalar('Upper Triangular Norm Losses', np.mean(norm_losses),epoch)

        if NET_TYPE in ['RORNN','RORNN2','ARORNN', 'ARORNN2']:
            writer.add_histogram('thetas', np.array([x.item() for x in net.rnn.thetas]),epoch)
            if NET_TYPE in ['ARORNN', 'ARORNN2']:
                writer.add_histogram('alphas', np.array([x.item() for x in net.rnn.alphas]),epoch)
                writer.add_scalar('Alpha Loss', np.mean(alpha_losses),epoch)

        #save data        
        if epoch % SAVEFREQ == 0 or epoch==num_epochs -1:
            with open(SAVEDIR + '{}_Train_Losses'.format(NET_TYPE), 'wb') as fp:
                pickle.dump(train_losses,fp)

            with open(SAVEDIR + '{}_Test_Losses'.format(NET_TYPE), 'wb') as fp:
                pickle.dump(test_losses,fp)

            with open(SAVEDIR + '{}_Test_Accuracy'.format(NET_TYPE),'wb') as fp:
                pickle.dump(test_accuracies,fp)
            
            with open(SAVEDIR + '{}_Train_Accuracy'.format(NET_TYPE),'wb') as fp:
                pickle.dump(train_accuracies,fp)
            with open(SAVEDIR + '{}_Grad_Norms'.format(NET_TYPE),'wb') as fp:
                pickle.dump(save_norms,fp)

            save_checkpoint({
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
                },
                '{}_{}.pth.tar'.format(NET_TYPE,epoch)
                )
        
    best_state = torch.load(os.path.join(SAVEDIR,'best_model.pth.tar'))
    net.load_state_dict(best_state['state_dict'])
    test_loss, test_acc = test_model(net,testloader)
    with open(os.path.join(SAVEDIR,'log_test.txt'), 'w') as fp:
        fp.write('Test loss: {} Test accuracy: {}'.format(test_loss,test_acc))

    return

lr = args.lr
lr_orth = args.lr_orth
random_seed = args.random_seed
NET_TYPE = args.net_type
CUDA = args.cuda
SAVEFREQ = args.save_freq
inp_size = 1
hid_size = args.nhid #calc_hidden_size(NET_TYPE,165000,1,10)
alam = args.alam
nonlins = ['relu','tanh','sigmoid','modrelu']
nonlin = args.nonlin.lower()
print(nonlin)
if nonlin not in nonlins:
    nonlin = 'none'
    print('Non lin not found, using no nonlinearity')
decay = args.weight_decay
udir = 'HS_{}_NL_{}_lr_{}_BS_{}_rinit_{}_iinit_{}_decay_{}_alpha_{}'.format(hid_size,nonlin,lr,args.batch,args.rinit,args.iinit, decay,args.alpha)
if NET_TYPE in ['RORNN2', 'ARORNN2', 'NRNN2', 'NSRNN2', 'EXPRNN']:
    udir += '_lro_{}'.format(args.lr_orth)
if NET_TYPE in ['ARORNN', 'ARORNN2']:
    udir += '_aL_{}'.format(alam)
LOGDIR = './logs/sMNIST/{}/{}/{}/'.format(NET_TYPE,udir,random_seed)
SAVEDIR = './saves/sMNIST/{}/{}/{}/'.format(NET_TYPE,udir,random_seed)
if not os.path.exists(SAVEDIR):
    os.makedirs(SAVEDIR)

with open(SAVEDIR + 'hparams.txt','w') as fp:
    for key,val in args.__dict__.items():
        fp.write(('{}: {}'.format(key,val)))

writer = SummaryWriter(LOGDIR)



T = 784
batch_size = args.batch
out_size = 10

rnn = select_network(NET_TYPE,inp_size,hid_size,nonlin,rinit,iinit,CUDA,args.ostep_method)

net = Model(hid_size,rnn)
if CUDA:
    net = net.cuda()
    net.rnn = net.rnn.cuda()
print('sMNIST task')
print(NET_TYPE)
print('Cuda: {}'.format(CUDA))
l2_norm_crit = nn.MSELoss()



orthog_optimizer = None
if args.ostep_method == 'exp' and NET_TYPE in ['ORNN2','ORNNR2','RORNN2','ARORNN2']:
    x = [
        {'params': (param for param in net.parameters() if param is not net.rnn.log_P and param is not net.rnn.P and param is not net.rnn.UppT)},
        {'params': net.rnn.UppT, 'weight_decay': decay}
        ]
    optimizer = optim.RMSprop(x, lr=args.lr,alpha=args.alpha)
    orthog_optimizer = optim.RMSprop([net.rnn.log_P],lr=args.lr_orth,alpha=args.alpha)
elif args.ostep_method == 'cayley' and NET_TYPE in ['ORNN2','ORNNR2','RORNN2','ARORNN2']:
    optimizer = optim.RMSprop((param for param in net.parameters()
                           if param is not net.rnn.P), lr=args.lr,alpha=args.alpha)
    orthog_optimizer = GeoSGD([net.rnn.P],lr=args.lr_orth,alpha=args.alpha)
elif args.ostep_method == 'exp' and NET_TYPE in ['NRNN2','NSRNN2']:
    x = [
        {'params': (param for param in net.parameters() if param is not net.rnn.log_Q and param is not net.rnn.Q and param is not net.rnn.UppT)},
        {'params': net.rnn.UppT, 'weight_decay': decay}
        ]
    optimizer = optim.RMSprop(x, lr=args.lr,alpha=args.alpha)
    orthog_optimizer = optim.RMSprop([net.rnn.log_Q],lr=args.lr_orth,alpha=args.alpha)
elif args.ostep_method == 'cayley' and NET_TYPE in ['NRNN2','NSRNN2']:
    x = [
        {'params': (param for param in net.parameters() if param is not net.rnn.log_Q and param is not net.rnn.Q and param is not net.rnn.UppT)},
        {'params': net.rnn.UppT, 'weight_decay': decay}
        ]
    optimizer = optim.RMSprop((param for param in net.parameters()
                           if param is not net.rnn.Q), lr=args.lr,alpha=args.alpha)
    orthog_optimizer = GeoSGD([net.rnn.Q],lr=args.lr_orth,alpha=args.alpha)
elif NET_TYPE == 'EXPRNN':
    optimizer = optim.RMSprop((param for param in net.parameters()
                           if param is not net.rnn.log_recurrent_kernel and 
                              param is not net.rnn.recurrent_kernel), lr=args.lr,alpha=args.alpha)
    orthog_optimizer = optim.RMSprop([net.rnn.log_recurrent_kernel],lr = args.lr_orth,alpha=args.alpha)
else:
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr,alpha=args.alpha)

epoch = 0

num_epochs = 70
train_model(net,optimizer,num_epochs)

