import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim
import numpy as np

class hr(nn.Module):

    def __init__(self,D,K,epoch=100,lr=0.01):
        #D: dimension
        #K: length of layer
        super(hr, self).__init__()
        self.D=D
        self.K=K
        self.epoch=epoch
        self.lr=lr
        self.optim_params=None
        self.loss_list=[]

        self.mean_nn=nn.Sequential(
            nn.Linear(self.D,self.K),
            nn.Sigmoid(),
            nn.Linear(self.K,1)
        )
        self.std_nn=nn.Sequential(
            nn.Linear(self.D,1),
            nn.Softplus()
        )
    
    def set_params(self,W1,b1,W2,b2,V1,c1):
        self.mean_nn[0].weight= nn.Parameter(torch.reshape(W1,torch.Size([self.K, self.D])))
        self.mean_nn[0].bias= nn.Parameter(torch.reshape(b1,torch.Size([self.K])))
        self.mean_nn[2].weight = nn.Parameter(torch.reshape(W2,torch.Size([1, self.K])))
        self.mean_nn[2].bias = nn.Parameter(torch.reshape(b2,torch.Size([1])))
        self.std_nn[0].weight = nn.Parameter(torch.reshape(V1,torch.Size([1, self.D])))
        self.std_nn[0].bias = nn.Parameter(torch.reshape(c1,torch.Size([1])))
        
        
    def mean(self,X):
        return self.mean_nn(X)
    
    def std(self,X):
        return self.std_nn(X)
    
    def nll(self,X,Y):
        means=self.mean(X)
        vars=self.std(X)**2
        ndist=torch.distributions.Normal(means,vars)
        nll=-ndist.log_prob(Y).sum()
        return nll
    
    def fit(self,X,Y,initialize_weights=True):
        if initialize_weights:
            self.mean_nn.apply(self.init_weights)
            self.std_nn.apply(self.init_weights)
            #self.print_param()
        optimizer=optim.Adam(self.parameters(),lr=self.lr)
        for epoch in range(self.epoch):
            loss=self.nll(X,Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.loss_list.append(loss.item())
            #print(f'Epoch:{epoch}   Loss:{loss.item()}')
        self.optim_params=self.parameters
        return loss.item()
    
    def init_weights(self,m):
        classname = m.__class__.__name__
        # for every Linear layer in a model
        if classname.find('Linear') != -1:
            y = m.in_features
            m.weight.data.normal_(0.0,1/np.sqrt(y))
            m.bias.data.normal_(0.0,0.1)
    
    def print_param(self):
        for param in self.parameters():
            print(f'parameter shape:{param.shape}')
            print(f'parameter value:{param.data}')