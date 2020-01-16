import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as T
import numpy as np
import pickle
import argparse
import time
import os
from utils import select_network, select_optimizer
from torch._utils import _accumulate
from torch.utils.data import Subset
from datetime import datetime

parser = argparse.ArgumentParser(description='auglang parameters')

parser.add_argument('--net-type', type=str, default='nnRNN',
                    choices=['RNN','nnRNN', 'LSTM', 'expRNN'],
                    help='options: RNN, nnRNN, expRNN, LSTM')
parser.add_argument('--nhid', type=int,
                    default=512,
                    help='hidden size of recurrent net')
parser.add_argument('--cuda', action='store_true',
                    default=False, help='use cuda')
parser.add_argument('--random-seed', type=int,
                    default=400, help='random seed')
parser.add_argument('--permute', action='store_true',
                    default=False, help='permute the order of sMNIST')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--save-freq', type=int,
                    default=50, help='frequency to save data')
parser.add_argument('--batch', type=int, default=100)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lr_orth', type=float, default=2e-5)
parser.add_argument('--optimizer',type=str, default='RMSprop',
                    choices=['Adam', 'RMSprop'],
                    help='optimizer: choices Adam and RMSprop')
parser.add_argument('--alpha',type=float,
                    default=0.99, help='alpha value for RMSprop')
parser.add_argument('--betas',type=tuple,
                    default=(0.9, 0.999), help='beta values for Adam')


parser.add_argument('--rinit', type=str, default="henaff",
                    choices=['random', 'cayley', 'henaff', 'xavier'],
                    help='recurrent weight matrix initialization')
parser.add_argument('--iinit', type=str, default="xavier",
                    choices=['xavier', 'kaiming'],
                    help='input weight matrix initialization' )
parser.add_argument('--nonlin', type=str, default='modrelu',
                    choices=['none','modrelu', 'tanh', 'relu', 'sigmoid'],
                    help='non linearity none, relu, tanh, sigmoid')
parser.add_argument('--alam', type=float, default=0.0001,
                    help='decay for gamma values nnRNN')
parser.add_argument('--Tdecay', type=float,
                    default=0, help='weight decay on upper T')





args = parser.parse_args()

torch.cuda.manual_seed(args.random_seed)
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

if args.permute:
    rng = np.random.RandomState(1234)
    order = rng.permutation(784)
else:
    order = np.arange(784)

trainset = T.datasets.MNIST(root='./MNIST',
                            train=True,
                            download=True,
                            transform=T.transforms.ToTensor())
valset = T.datasets.MNIST(root='./MNIST',
                          train=True,
                          download=True,
                          transform=T.transforms.ToTensor())
offset = 10000
R = rng.permutation(len(trainset))
lengths = (len(trainset) - offset, offset)
trainset,valset = [Subset(trainset, R[offset - length:offset])
                   for offset, length in zip(_accumulate(lengths), lengths)]
testset = T.datasets.MNIST(root='./MNIST',
                           train=False,
                           download=True,
                           transform=T.transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch,
                                          shuffle=False,
                                          num_workers=2)
valloader = torch.utils.data.DataLoader(valset,
                                        batch_size=args.batch,
                                        shuffle=False,
                                        num_workers=2)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=args.batch,
                                         num_workers=2)


class Model(nn.Module):
    def __init__(self, hidden_size, rnn):
        super(Model, self).__init__()
        self.rnn = rnn
        self.hidden_size = hidden_size
        self.lin = nn.Linear(hidden_size, 10)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, inputs, y, order):
        h = None
        inputs = inputs[:, order]
        for input in torch.unbind(inputs, dim=1):
            h = self.rnn(input.unsqueeze(1), h)
        out = self.lin(h)
        
        loss = self.loss_func(out, y)
        preds = torch.argmax(out, dim=1)
        correct = torch.eq(preds, y).sum().item()
        return loss, correct

def test_model(net, dataloader):
    accuracy = 0
    loss = 0
    net.eval()
    with torch.no_grad():   
        for i, data in enumerate(dataloader):
            x,y = data
            x = x.view(-1, 784)
            if CUDA:
                x = x.cuda()
                y = y.cuda()
            if NET_TYPE == 'LSTM':
                net.rnn.init_states(x.shape[0])
            loss,c = net.forward(x, y, order)
            accuracy += c
    accuracy /= len(testset)
    return loss, accuracy

def save_checkpoint(state, fname):
    filename = os.path.join(SAVEDIR, fname)
    torch.save(state, filename)

def train_model(net, optimizer, num_epochs):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    best_test_acc = 0

    for epoch in range(0, num_epochs):
        s_t = time.time()
        accs = []
        losses = []
        processed = 0
        alpha_losses = []
        net.train()
        correct = 0
        for i,data in enumerate(trainloader, 0):
            inp_x, inp_y = data
            inp_x = inp_x.view(-1, 784)

            if CUDA:
                inp_x = inp_x.cuda()
                inp_y = inp_y.cuda()
            
            if NET_TYPE == 'LSTM':
                net.rnn.init_states(inp_x.shape[0])

            optimizer.zero_grad()
            if orthog_optimizer:
                orthog_optimizer.zero_grad()
            
            loss, c = net.forward(inp_x, inp_y, order)
            correct += c
            processed += inp_x.shape[0]

            accs.append(correct/float(processed))
            
            #calculate losses for orthogonal rnn and alpha blocks
            if NET_TYPE == 'nnRNN' and alam > 0:
                alpha_loss = net.rnn.alpha_loss(alam)
                loss += alpha_loss
                alpha_losses.append(alpha_loss.item())

            loss.backward()
            losses.append(loss.item())
            if orthog_optimizer:
                net.rnn.orthogonal_step(orthog_optimizer)
            optimizer.step()
        
        test_loss, test_acc = test_model(net, valloader)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_checkpoint({
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
                },
                '{}.pth.tar'.format('best_model')
                )
   

        print('Epoch {}, Time for Epoch: {}, Train Loss: {}, '
              'Train Accuracy: {} Test Loss: {} Test Accuracy {}'
              .format(epoch +1, time.time()- s_t, np.mean(losses),
                      np.mean(accs), test_loss, test_acc))
        train_losses.append(np.mean(losses))
        train_accuracies.append(np.mean(accs))

        #save data
        if epoch % SAVEFREQ == 0 or epoch==num_epochs -1:
            with open(SAVEDIR + '{}_Train_Losses'.format(NET_TYPE),
                      'wb') as fp:
                pickle.dump(train_losses, fp)

            with open(SAVEDIR + '{}_Test_Losses'.format(NET_TYPE),
                      'wb') as fp:
                pickle.dump(test_losses, fp)

            with open(SAVEDIR + '{}_Test_Accuracy'.format(NET_TYPE),
                      'wb') as fp:
                pickle.dump(test_accuracies, fp)

            with open(SAVEDIR + '{}_Train_Accuracy'.format(NET_TYPE),
                      'wb') as fp:
                pickle.dump(train_accuracies, fp)

            save_checkpoint({
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
                },
                '{}_{}.pth.tar'.format(NET_TYPE,epoch)
                )
        
    best_state = torch.load(os.path.join(SAVEDIR, 'best_model.pth.tar'))
    net.load_state_dict(best_state['state_dict'])
    test_loss, test_acc = test_model(net, testloader)
    with open(os.path.join(SAVEDIR, 'log_test.txt'), 'w') as fp:
        fp.write('Test loss: {} Test accuracy: {}'.format(test_loss, test_acc))

    return

lr = args.lr
lr_orth = args.lr_orth
random_seed = args.random_seed
NET_TYPE = args.net_type
CUDA = args.cuda
SAVEFREQ = args.save_freq
inp_size = 1
hid_size = args.nhid
alam = args.alam
Tdecay = args.Tdecay

exp_time = "{0:%Y-%m-%d}_{0:%H-%M-%S}".format(datetime.now())
SAVEDIR = os.path.join('./saves',
                       'sMNIST',
                       NET_TYPE,
                       str(random_seed),
                       exp_time)

if not os.path.exists(SAVEDIR):
    os.makedirs(SAVEDIR)

with open(SAVEDIR + 'hparams.txt','w') as fp:
    for key,val in args.__dict__.items():
        fp.write(('{}: {}'.format(key,val)))

T = 784
batch_size = args.batch
out_size = 10

rnn = select_network(args, inp_size)

net = Model(hid_size,rnn)
if CUDA:
    net = net.cuda()
    net.rnn = net.rnn.cuda()
print('sMNIST task')
print(NET_TYPE)
print('Cuda: {}'.format(CUDA))
optimizer, orthog_optimizer = select_optimizer(net, args)
epoch = 0
num_epochs = args.epochs
train_model(net, optimizer, num_epochs)

