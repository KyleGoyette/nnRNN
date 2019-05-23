import pickle
import os

class DataSaver:
    def __init__(self,save_dir):
        self.forward = {}
        self.net = {}
        self.backward = {}
        self.save_dir = save_dir
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
    '''
    '''
    def store_forward(self,epoch,t,inputs,s_hidden,hidden,output,W, U,V,bias_h,bias_o):
        t = int(t)
        self.epoch = epoch
        if epoch not in self.forward.keys():
            self.forward[epoch] = {}
        if t not in self.forward[epoch].keys():
            self.forward[epoch][t] = {}
        self.forward[epoch][t]['inputs'] = inputs.cpu().detach().numpy()
        self.forward[epoch][t]['s_hidden'] = s_hidden.cpu().detach().numpy()
        self.forward[epoch][t]['hidden'] = hidden.cpu().detach().numpy()
        self.forward[epoch][t]['outputs'] = output.cpu().detach().numpy()
        if epoch not in self.net.keys():
            self.net[epoch] = {}
        self.net[epoch]['V'] = V.cpu().detach().numpy()
        self.net[epoch]['U'] = U.cpu().detach().numpy()
        self.net[epoch]['W'] = W.cpu().detach().numpy()
        self.net[epoch]['bias_o'] = bias_o.cpu().detach().numpy()
        self.net[epoch]['bias_h'] = bias_h.cpu().detach().numpy()

    '''
    '''
    def store_grads(self,t,grad_hidden_o, grad_hidden_h, grad_W, grad_V, grad_U,grad_bias_o, grad_bias_h, grad_hidden_in):
        epoch = self.epoch
        t = int(t)
        if epoch not in self.backward.keys():
            self.backward[epoch] = {}
        if t not in self.backward[epoch].keys():
            self.backward[epoch][t] = {}
        self.backward[epoch][t]['grad_hidden_o'] = grad_hidden_o.cpu().detach().numpy()
        self.backward[epoch][t]['grad_hidden_h'] = grad_hidden_h.cpu().detach().numpy()
        self.backward[epoch][t]['grad_hidden_in'] = grad_hidden_in.cpu().detach().numpy()
        self.backward[epoch][t]['grad_W'] = grad_W.cpu().detach().numpy()
        self.backward[epoch][t]['grad_V'] = grad_V.cpu().detach().numpy()
        self.backward[epoch][t]['grad_U'] = grad_U.cpu().detach().numpy()
        self.backward[epoch][t]['grad_bias_o'] = grad_bias_o.cpu().detach().numpy()
        self.backward[epoch][t]['grad_bias_h'] = grad_bias_h.cpu().detach().numpy()
    
    def save_data(self,fname):
        pickle.dump(self.net,open(self.save_dir + fname + '_net.pkl','wb'))
        pickle.dump(self.forward,open(self.save_dir + fname + '_forward.pkl','wb'))
        pickle.dump(self.backward,open(self.save_dir + fname + '_backward.pkl','wb'))