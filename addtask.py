import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from LSTM import LSTM
from RNN import RNN
from GRU import GRU
import matplotlib.pyplot as plt
import os
import pickle
import argparse
from tensorboardX import SummaryWriter
import shutil

import glob
import os

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='auglang parameters')

parser.add_argument('--net-type', type=str, default='LSTM', help='recurrent network type')
parser.add_argument('--s-detach', type=str2bool, default=False, help='controls h-detach type from all to singular')
parser.add_argument('--p-detach', type=float, default=1.0, help='probability of detaching each timestep')
parser.add_argument('--hidden-size', type=int, default=128, help='hidden size of recurrent net')
parser.add_argument('--log-dir', type=str, default='./logs/addtask/', help='log dir of the results')
parser.add_argument('--save-dir', type=str, default='./saves/addtask/', help='save dir of net and LE results')
parser.add_argument('--LED-freq', type=int, default=5, help='frequency to measure LE Dynamics')
parser.add_argument('--save-freq', type=int, default=50, help='frequency to save data')
parser.add_argument('--cuda', type=str2bool, default=True, help='use cuda')
parser.add_argument('--T', type=int, default=150, help='sequence length')
parser.add_argument('--random-seed', type=int, default=400, help='random seed')
parser.add_argument('--train', type=str2bool, default=True, help='train a model')
parser.add_argument('--load-model', type=str, default=None, help='model path to load')
parser.add_argument('--l2-reg', type=float, default=0.0, help='l2-regularization parameter')


args = parser.parse_args()


def generate_adding_sequence(T):

    x = []
    sq = np.random.uniform(size=T)

    x = np.zeros((2*T,1))

    x[:T,0] = sq
    fv = np.random.randint(0,T//2,1)[0]
    sv = np.random.randint(T//2,T,1)[0]

    x[T+fv] = 1.0
    x[T+sv] = 1.0
    

    y = torch.FloatTensor(np.array(sq[fv] + sq[sv]))
    x = torch.FloatTensor(x)
    return x,y

def create_dataset(size, T):
    d_x = []
    d_y = []
    for i in range(size):
        sq_x, sq_y = generate_adding_sequence(T)
        #sq_x, sq_y = sq_x[0], sq_y[0]
        d_x.append(sq_x)
        d_y.append(sq_y)

    d_x = torch.stack(d_x)
    d_y = torch.stack(d_y)
    return d_x, d_y

class Net(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,rec_net):
        super(Net,self).__init__()
        self.rec_net = rec_net
        self.ol = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x, new_state = self.rec_net.forward(x)
        out = self.ol(x)
        return out, new_state


def forward_prop(model,x,y):
    for i in range(2*T):
        if p_detach != 1.0 and NET_TYPE=='LSTM' and not singular:
            rand_val = np.random.random(size=1)[0]
            if rand_val <= p_detach:
                model.rec_net.hidden = model.rec_net.hidden.detach()
        elif p_detach != 1.0 and NET_TYPE == 'LSTM' and singular:
            a = p_detach*torch.ones(model.rec_net.hidden.shape)
            b = torch.bernoulli(a)
            if CUDA:
                b = b.cuda()
            z = model.rec_net.hidden.detach()
            model.rec_net.hidden = torch.where(b != 0,model.rec_net.hidden, z)
        output,_ = model(x[i])
    loss = crit(output,y.unsqueeze(1))

    return loss


def oneStepVarQR(Q,M):
    Z = np.dot(M,Q)
    q,r = np.linalg.qr(Z,mode='reduced')
    s = np.diag(np.sign(np.diag(r)))
    #print(np.diag(s))
    return q.dot(s), np.diag(r.dot(s))

def calc_LEs_and_rs(net,x,l,i,j,num_LEs):
    
    num_steps = x.shape[0]
    x = x.unsqueeze(1)
    if CUDA:
        x = x.cuda()
    net.rec_net.init_hidden(1)
    Q = np.eye(num_LEs,num_LEs)
    q_vect = []
    rvals = []
    #traj = np.ndarray((num_steps,net.rec_net.hidden_size))
    for ind in range(x.shape[0]):
        ht, _ = net.rec_net.forward(x[ind])
        M_t = net.rec_net.construct_M(l,i,j,torch.cat((x[ind],ht),1))
        M_t = M_t.cpu().detach().numpy()
        Q, r = oneStepVarQR(Q,M_t)
        #traj[i,:] = ht.data
        q_vect.append(Q)
        rvals.append(r)
    rvals = np.vstack(rvals)
    LEs = np.sum(np.log2(rvals),axis=0)/num_steps
    return LEs, rvals, q_vect

def show_traj(net,x):
    x = x.transpose(0,1)
    print(x.shape)
    net.rec_net.init_hidden(1)
    traj = np.ndarray((x.shape[0],net.rec_net.hidden_size))
    for i in range(x.shape[0]):
        
        h, _ = net.rec_net.forward(x[i,0].unsqueeze(1))
        traj[i,:] = h.data

    return traj

def test_model(net,test_x, test_y):
    net.rec_net.init_hidden(test_x.shape[0])
    if CUDA:
        test_x = test_x.cuda()
        test_y = test_y.cuda()
    test_x = test_x.transpose(0,1)

    with torch.no_grad():
        for i in range(2*T):
            output, _ = net.forward(test_x[i])
        loss = crit(output,test_y.unsqueeze(1)).item()

    accuracy = torch.mean(torch.div(torch.abs(output - test_y.unsqueeze(1)),test_y.unsqueeze(1)),dim=0)
    print(accuracy)
    return loss



def train_model(net,num_epochs,batch_size,train_size,optimizer,num_LEs,test_size):
    if not os.path.exists(SAVEDIR):
        os.makedirs(SAVEDIR)
    logdir = LOGDIR + '{}_{}_{}_{}/'.format(NET_TYPE,p_detach,singular,random_seed)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = SummaryWriter(log_dir=logdir)

    train_x, train_y = create_dataset(train_size,T)
    test_x, test_y = create_dataset(test_size,T)
    
    all_LEs = []
    all_eigs = []
    all_losses = []
    all_accs = []
    test_losses = []
    all_Qs = []
    inds = np.arange(train_size)
    for epoch in range(num_epochs):
        losses = []
        np.random.shuffle(inds)
        for i in range(int(train_size/batch_size)):
            ind = inds[i*(batch_size):(i+1)*batch_size]
            x = train_x[ind]
            y = train_y[ind]
            if i == 1 and epoch % LED_FREQ == 0:
                LEs, rs, q_vect = calc_LEs_and_rs(net,x[i,:],'o',0,1,num_LEs)
                e,v = np.linalg.eig(net.rec_net.Wo.weight.data[:hid_size,:hid_size])
                all_LEs.append(LEs)
                all_eigs.append(e)
                all_Qs.append(q_vect)
            if CUDA:
                x = x.cuda()
                y = y.cuda()
            x = x.transpose(0,1)
            net.zero_grad()
            net.rec_net.init_hidden(x.shape[1])
            loss = forward_prop(net,x,y)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        all_losses.append(sum(losses)/(train_size/batch_size))


        test_loss = test_model(net,test_x,test_y)
        test_losses.append(test_loss)

        writer.add_scalar('Train Losses',sum(losses),epoch)
        writer.add_scalar('Test Loss',test_loss,epoch)        

        print('Epoch {}, Average Loss: {}, Test Loss: {}'.format(epoch +1, sum(losses)/(train_size/batch_size), test_loss))
    
        if epoch % SAVEFREQ == 0 or epoch == num_epochs -1 :
            with open(SAVEDIR + '{}_{}_{}_losses'.format(NET_TYPE,p_detach,singular), 'wb') as fp:
                pickle.dump(all_losses,fp)

            with open(SAVEDIR + '{}_{}_{}_LEs'.format(NET_TYPE,p_detach,singular), 'wb') as fp:
                pickle.dump(all_LEs,fp)

            with open(SAVEDIR + '{}_{}_{}_Eigs'.format(NET_TYPE,p_detach,singular),'wb') as fp:
                pickle.dump(all_eigs,fp)

            with open(SAVEDIR + '{}_{}_{}_TestLosses'.format(NET_TYPE,p_detach,singular),'wb') as fp:
                pickle.dump(test_losses,fp)

            save_checkpoint({
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()
                },
                test_loss == min(test_losses),
                '{}_{}_{}_{}.pth.tar'.format(NET_TYPE,p_detach,singular,epoch)
                )
        

def load_model(net, fname):
    if fname == 'y':
        list_of_files = glob.glob('./saves/{}_{}_{}*.pth.tar'.format(NET_TYPE,p_detach,singular)) # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        #latest_file = './saves/LSTM_1.0_False_350.pth.tar'
        print('Loading {}'.format(latest_file))

        #latest_file = './saves/LSTM_1.0_False_350.pth.tar'
        check = torch.load(latest_file)
        net.load_state_dict(check['state_dict'])
    else:
        check = torch.load(fname)
        net.load_state_dict(check['state_dict'])
    
    return net

def save_checkpoint(state,is_best,fname):
    filename = SAVEDIR + fname
    torch.save(state,filename)
    if is_best:
        shutil.copyfile(filename, SAVEDIR + 'model_best.pth.tar')

random_seed = args.random_seed
NET_TYPE = args.net_type
LED_FREQ = args.LED_freq 
p_detach = args.p_detach
singular = args.s_detach
CUDA = args.cuda
LOGDIR = args.log_dir
SAVEDIR = args.save_dir + str(random_seed) + '/'
SAVEFREQ = args.save_freq
loadmodel = args.load_model
train = args.train
l2_reg = args.l2_reg

torch.cuda.manual_seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)





inp_size = 1
hid_size = args.hidden_size
T = args.T
batch_size = 100
out_size = 1
num_LEs = 2*hid_size

if NET_TYPE == 'LSTM':
    rec_net = LSTM(inp_size,hid_size,p_detach,CUDA)
    num_LEs = 3*hid_size
elif NET_TYPE == 'RNN':
    rec_net = RNN(inp_size,hid_size,CUDA)
elif NET_TYPE == 'GRU':
    rec_net = GRU(inp_size,hid_size,CUDA)
net = Net(inp_size,hid_size,out_size,rec_net)
if CUDA:
    net = net.cuda()

print(CUDA)
print(NET_TYPE)
print(p_detach)
print(singular)
#print(net)


if load_model != None:
    net = load_model(net,loadmodel)

if train:
    num_epochs = 600


    train_size = 1000
    test_size = 500
    lr = 0.0001
    crit = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=l2_reg)
    test_x, test_y = create_dataset(test_size,T)


    train_model(net,num_epochs,batch_size,train_size,optimizer,num_LEs,test_size)

else:
    test_size = 5
    crit = nn.MSELoss()
    test_x, test_y = create_dataset(test_size,T)
    print(test_x)
    accs = []

    loss = test_model(net,test_x, test_y)
    #accs.append(acc)
    print(loss)
    


    _,_, q_vect = calc_LEs_and_rs(net,test_x[0,:],'o',0,1,num_LEs)
    iteration = 10
    Q = q_vect[iteration]
    n = Q.shape[0]
    for node_choice in range(hid_size):
        v = np.zeros(n)
        v[node_choice] = 1
        w = np.zeros(n)
        subspaces = np.zeros(n)
        for k in range(n):
            w += v.dot(Q[:,k])*Q[:,k]
            subspaces[k] = np.sqrt(w.dot(w))

        plt.plot(subspaces, '--',markersize=6)
    plt.plot([1,n],[0,1],'k-')
    plt.title('{} p-detach = {}, detach_type = {},Node Participation for iteration {}'.format(NET_TYPE, p_detach, singular,iteration))
    plt.xlabel('cumulative Lyap Subspace')
    plt.ylabel('norm of projection')
    plt.show()

    traj = show_traj(net,test_x)

    traj_plot = plt.figure()
    for n in range(max(hid_size,10)):
        plt.plot(range(2*T),traj[:,n])
    plt.ylabel('activation')
    plt.xlabel('')
    plt.title('Activation over input sequence, {}, {}, {}'.format(NET_TYPE,p_detach, singular))
    plt.show()