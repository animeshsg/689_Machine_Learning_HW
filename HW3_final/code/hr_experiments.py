import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from hr import hr

#Function for plotting model output
def make_hr_plot(model,Xtr,Ytr,filename=None):
    
    x=np.linspace(0,1,100)
    x=x.reshape((100,1))
    m=model.mean(torch.tensor(x)).detach().numpy().flatten()
    s=model.std(torch.tensor(x)).detach().numpy().flatten()

    plt.plot(Xtr,Ytr,"k.")
    plt.plot(x.flatten(),m,'b' )
    plt.fill_between(x.flatten(),m-2*s,m+2*s,alpha=0.5)
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(["Data","Mean","2 Std"])
    if filename is not None:
        plt.savefig(filename)
    plt.show() 

#Code implementing experiments
if __name__ == "__main__":

    #Load the data
    data = np.load("../data/hr_data.npz")
    Xtr=torch.tensor(data["Xtr"])
    Xte=torch.tensor(data["Xte"])
    Ytr=torch.tensor(data["Ytr"])
    Yte=torch.tensor(data["Yte"])

    #Load the parameters
    params = np.load("../data/hr_params.npz")
    W1=torch.tensor(params["W1"])
    b1=torch.tensor(params["b1"])
    W2=torch.tensor(params["W2"])
    b2=torch.tensor(params["b2"])
    V1=torch.tensor(params["V1"])
    c1=torch.tensor(params["c1"])

    #Create the model:
    model=hr(1,3)

    #Show model
    make_hr_plot(model,Xtr,Ytr)

    #a
    model.set_params(W1,b1,W2,b2,V1,c1)
    means=model.mean(Xtr)
    print(means[0:5])

    #b
    model.set_params(W1,b1,W2,b2,V1,c1)
    stds=model.std(Xtr)
    print(stds[0:5])

    #c
    model.set_params(W1,b1,W2,b2,V1,c1)
    nll=model.nll(Xtr,Ytr)
    print(nll.item())

    #d
    epochs=[50,100,500,1000,1500,2000,2500,5000]
    lrs=[1,1e-1,1e-2,1e-3,1e-4,1e-5]
    loss_epoch=[]
    for epoch in epochs:
        loss_lr=[]
        for lr in lrs:
            model=hr(1,5,epoch,lr)
            loss=model.fit(Xtr,Ytr,False)
            loss_lr.append(loss)
            print(f'Epoch:{epoch}   lr: {lr}    Loss:{loss}')
        loss_epoch.append(loss_lr)
    print(loss_epoch)

    #e
    optimal_epoch=1500
    optimal_lr=0.1
    loss_lists_tr=[]
    loss_lists_te=[]
    model_list_tr=[]
    for i in range(5):
        model_tr=hr(1,5,optimal_epoch,optimal_lr)
        loss_tr=model_tr.fit(Xtr,Ytr)
        model_list_tr.append(model_tr)
        loss_lists_tr.append(model_tr.loss_list)
        model_te=hr(1,5,optimal_epoch,optimal_lr)
        loss_te=model_te.fit(Xte,Yte)
        loss_lists_te.append(model_te.loss_list)
    
    values = [i for i in range(optimal_epoch)]

    for i in range(len(loss_lists_tr)):
        plt.plot(values, loss_lists_tr[i], label=f"random_initialization={i}")

    plt.xlabel('Epoch (log scale)')
    plt.ylabel('negative log likelihood')
    plt.title('Training NLL Loss vs. Epoch')
    plt.legend()
    plt.xscale('log')
    plt.show()

    values = [i for i in range(optimal_epoch)]

    for i in range(len(loss_lists_te)):
        plt.plot(values, loss_lists_te[i], label=f"random_initialization={i}")

    plt.xlabel('Epoch (log scale)')
    plt.ylabel('negative log likelihood')
    plt.title('Test NLL Loss vs. Epoch')
    plt.legend()
    plt.xscale('log')
    plt.show()

    #f
    make_hr_plot(model_list_tr[4],Xtr,Ytr)


