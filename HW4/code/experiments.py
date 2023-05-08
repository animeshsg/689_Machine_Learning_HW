import numpy as np
import torch
import torch.optim as optim

np.random.seed(1)
torch.manual_seed(1)
torch.set_default_dtype(torch.float64)

data = np.load("../data/data.npz")
X_tr=torch.tensor(data['X_train'])
Y_tr=torch.tensor(data['Y_train'])
X_te=torch.tensor(data['X_test'])
Y_te=torch.tensor(data['Y_test'])
        

