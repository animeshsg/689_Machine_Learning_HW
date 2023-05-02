import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ae_models_mlp import MLP, train_mlp, grid_search_mlp
#from ae_MLP import RNN

torch.manual_seed(1)

data = np.load("../data/ae_data.npz")

Str = data["Str"]
Xtr = torch.from_numpy(data["Xtr"]).type(torch.int)
Ytr = torch.from_numpy(data["Ytr"]).type(torch.float)

Ste = data["Ste"]
Xte = torch.from_numpy(data["Xte"]).type(torch.int)
Yte = torch.from_numpy(data["Yte"]).type(torch.float)

#print("Example data cases:")
#for i in range(10):
#    print("%d: (%s,%d)"%(i,Str[i],Ytr[i]))

# Set hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 100


#model = MLP()
#criterion = nn.MSELoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


#model = LSTM()
#criterion = nn.MSELoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


#model = Transformer()
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


torch.manual_seed(1)

data = np.load("../data/ae_data.npz")

Str = data["Str"]
Xtr = torch.from_numpy(data["Xtr"]).type(torch.int)
Ytr = torch.from_numpy(data["Ytr"]).type(torch.float)

Ste = data["Ste"]
Xte = torch.from_numpy(data["Xte"]).type(torch.int)
Yte = torch.from_numpy(data["Yte"]).type(torch.float)

# set random seed for reproducibility
np.random.seed(42)

# randomly shuffle the data indices
indices = np.arange(Xtr.shape[0])
np.random.shuffle(indices)

# define validation set size
val_size = 0.2  # 20% of training set

# split the shuffled data into training and validation sets
split = int(np.floor(Xtr.shape[0] * (1 - val_size)))
train_idx, val_idx = indices[:split], indices[split:]
Xtrain, Ytrain = Xtr[train_idx], Ytr[train_idx]
Xval, Yval = Xtr[val_idx], Ytr[val_idx]

# Define model and optimizer
model = MLP(input_dim=Xtrain.shape[1], hidden_dim=64, output_dim=1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
#train_mlp(Xtrain, Ytrain, model, optimizer)

learning_rates = [0.1, 0.01, 0.001]
hidden_sizes = [64, 128, 256]
batch_sizes = [128, 256, 512, 1024]

grid_search_mlp(Xtrain, Ytrain, Xval, Yval, learning_rates, hidden_sizes, batch_sizes)
