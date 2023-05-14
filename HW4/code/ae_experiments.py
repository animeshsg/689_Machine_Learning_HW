import numpy as np
import torch
import torch.nn as nn
import  matplotlib.pyplot as plt
from ae import linear_AE,Relu_AE,leakyRelu_AE,tanh_AE,mlpRelu_AE,RNNAE
from sklearn.model_selection import train_test_split

np.random.seed(17)
torch.manual_seed(17)
torch.set_default_dtype(torch.float)
device = torch.device('mps' if torch.backends.mps.is_available() else 'mps')

data       = np.load("../data/ecg_data.npz")
Xtr        = torch.tensor(data["Xtr"]).type(torch.float).to(device)    #Clean train
Xtr_noise  = torch.tensor(data["Xtr_noise"]).type(torch.float).to(device)  #Noisy train
Xte_noise  = torch.tensor(data["Xte_noise"]).type(torch.float).to(device)  #Noisy test

params = np.load("../data/ecg_params.npz")
W = torch.FloatTensor(params["W"]).type(torch.float).to(device)  #Decoder parameters
V = torch.FloatTensor(params["V"]).type(torch.float).to(device)  #Encoder parameters

#a
model=linear_AE(100,5,lr=0.001).to(device)
X_hat=model.reconstruct(Xte_noise[:5],V,W)
for i in range(5):
    model.plot_loss(Xte_noise[i],X_hat[i])

#b
X_hat=model.reconstruct(Xtr_noise[:5],V,W)
bloss=model.loss(Xtr_noise[:5],X_hat)
print("b:",bloss)

#c
model1=linear_AE(100,10,lr=0.01,epoch=2000).to(device)
loss=model1.fit(Xtr,Xtr)
print("c1:",loss)
x_pred_noisy=model1.forward(Xtr_noise)
print("c2:",model1.loss(Xtr,x_pred_noisy))

#d
model1=linear_AE(100,10,lr=0.01,epoch=2000).to(device)
loss=model1.fit(Xtr_noise,Xtr)
print("d:",loss)

#f
K=[k for k in range(10,150,5)]
train_loss_list=[]
test_loss_list=[]
best_k=0
best_loss=np.inf
best_model=None
X_train, X_val, R_train, R_val = train_test_split(Xtr_noise, Xtr, test_size=0.2)
for k in K:
    model=linear_AE(100,k,lr=0.001,epoch=1500).to(device)
    train_loss=model.fit(X_train,R_train)
    train_loss_list.append(train_loss)
    test_loss=model.test(X_val,R_val)
    test_loss_list.append(test_loss)
    if test_loss<best_loss:
        best_k=k
        best_model=model
        best_loss=test_loss
    print(f'K:{k} Train Loss:{train_loss}   Test Loss:{test_loss}')

print(f'Best K:{best_k}   Best Loss:{best_loss}')

test_loss_lists=[x.cpu().detach().numpy() for x in test_loss_list]
plt.plot(K, train_loss_list, label='train loss')
plt.plot(K, test_loss_lists, label='test loss')
plt.xlabel('K')
plt.ylabel('loss')
plt.title('K Vs loss (Linear Model)')
plt.legend()
plt.show()

#h
best_model=mlpRelu_AE(100,120,lr=0.001,epoch=1500).to(device)
train_loss=best_model.fit(Xtr_noise,Xtr)
X_pred=best_model.forward(Xte_noise)
for i in range(5):
    best_model.plot_loss(Xte_noise[i],X_pred[i])
