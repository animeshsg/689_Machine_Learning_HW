import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ae_models import RNNModel,TransformerModel,MLPModel,HelperClass

torch.manual_seed(17)

data = np.load("../data/ae_data.npz")

Str = data["Str"]
Xtr = torch.from_numpy(data["Xtr"]).type(torch.int)
Ytr = torch.from_numpy(data["Ytr"]).type(torch.float)

Ste = data["Ste"]
Xte = torch.from_numpy(data["Xte"]).type(torch.int)
Yte = torch.from_numpy(data["Yte"]).type(torch.float)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
torch.manual_seed(1)
print(device)


num_epochs = 1000

##MLP grid search
hidden_layers=[128,256,512,1024]
mlp_best_loss=float("inf")
best_mlp_model=None
for hidden_layer in hidden_layers:
    model = MLPModel(7, hidden_layer, 1).to(device)
    mlp=HelperClass(Xtr,Ytr,model,num_epochs)
    mlp_train_loss=mlp.trainer()
    if mlp_train_loss<mlp_best_loss:
        best_mlp_model=model
        mlp_best_loss=mlp_train_loss
print(best_mlp_model.__repr__())
print(mlp_best_loss)


hidden_sizes = [32,64,128,256]
num_layers = [2,4,8]
embedding_size=16
rnn_best_loss=float("inf")
best_rnn_model=None
for hidden_size in hidden_sizes:
    for num_layer in num_layers:

        model = RNNModel(input_size=13, hidden_size=hidden_size, num_layers=num_layers, output_size=1,embedding_size=embedding_size).to(device)
        rnn=HelperClass(Xtr,Ytr,model,num_epochs)
        rnn_train_loss=rnn.trainer()
        if rnn_train_loss<rnn_best_loss:
            best_rnn_model=model
            rnn_best_loss=rnn_train_loss
print(best_rnn_model.__repr__())
print(mlp_best_loss)

output_vocab_size=1
num_features=7
d_models=[16,32,64,128]
nheads=[2,4,8]
num_layers=[2,4,8]
trans_best_loss=float("inf")
best_trans_model=None
for d_model in d_models:
    for nhead in nheads:
        for num_layer in num_layers:
            model = TransformerModel(num_features,d_model,nhead,num_layer,output_vocab_size).to(device)
            trans=HelperClass(Xtr,Ytr,model,num_epochs)
            trans_train_loss=trans.trainer()
            if trans_train_loss<trans_best_loss:
                best_trans_model=model
                trans_best_loss=trans_train_loss
print(best_trans_model.__repr__())
print(trans_best_loss)


# best model = RNN
# with hyperparameters
# num_epochs = 1000
# batch_size = 1024
# learning_rate = 1e-5
# hidden_size = 64
# num_layers = 4
# embedding_size=16



