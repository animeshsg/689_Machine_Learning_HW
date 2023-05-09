import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import copy
import torch.nn.functional as F

class ArithmeticExpressionDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return x, y

class MLPModel(nn.Module):

    def __repr__(self):
        return {"model":"MLP","FL":self.input_dim,"HL":self.hidden_dim}
    
    def __init__(self, input_dimension, hidden_dimension, output_dimension):
        super(MLPModel, self).__init__()
        self.input_dim = input_dimension
        self.hidden_dim = hidden_dimension
        self.output_dim = output_dimension
        self.layers=nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, output_dimension)
        )

    def forward(self, x):
        x=x.to(torch.float32)
        x = self.layers(x)
        return x

class RNNModel(nn.Module):

    def __repr__(self):
        return {"model":"RNN","ES":self.embedding_size,"HS":self.hidden_size}
    
    def __init__(self, input_size, hidden_size, output_size, num_layers, embedding_size):
        super(RNNModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        #Model Arch
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, batch_first=True,nonlinearity="relu")
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])

        return out

class TransformerModel(nn.Module):

    def __repr__(self):
        return {"model":"Transformer","DM":self.d_model,"NH":self.nhead,"NL":self.num_layers}
    
    
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.d_model=d_model
        self.nhead=nhead
        self.num_layers=num_layers

        self.embedding = nn.Embedding(input_dim, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers)
        self.output = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        out = self.transformer(x, x)
        out = self.output(out[-1, :, :])
        return out
    
class HelperClass:
    def __init__(self,X,Y,model,epochs,lr=1e-3):
        self.model=model
        self.epochs=epochs
        self.optimizer=optim.Adam(self.model.parameters(), lr=lr)
        self.X=X
        self.Y=Y
        self.criterion=nn.MSELoss()
    
    @staticmethod
    def init_device():
        device = torch.device('mps' if torch.backends.mps.is_available() else 'mps')
        torch.manual_seed(1)
        return device
    
    @staticmethod
    def plot_loss(train_loss, val_loss, epochs):
        plt.plot(range(1, epochs+1), train_loss, label='Training Loss')
        plt.plot(range(1, epochs+1), val_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()
    
    def dataloader(self,X,Y,batch_size=1024):
        #X_train, X_val, Y_train, Y_val = train_test_split(Xtr, Ytr, test_size=0.2)
        dataset = ArithmeticExpressionDataset(X,Y)
        dataloader=DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def saver(self,best_weights,PATH="./models"):
        name=''.join(f"{key}{val}" for key, val in self.model.__repr__().items())
        path=PATH+"/"+name
        torch.save(best_weights,path)

    def trainer(self):
        device=self.init_device()
        val_loss_list=[]
        train_loss_list=[]
        X_train, X_val, Y_train, Y_val = train_test_split(self.X, self.Y, test_size=0.2)
        train_loader=self.dataloader(X_train,Y_train)
        val_loader=self.dataloader(X_val,Y_val)
        best_valid_loss = float("inf")
        for epoch in range(self.epochs):
            train_loss = 0.0
            val_loss = 0.0
            self.model.train()
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * x.size(0)
            train_loss /= len(train_loader.dataset)
            train_loss_list.append(train_loss)
            self.model.eval()
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    output = self.model(x)
                    loss = self.criterion(output, y)
                    val_loss += loss.item() * x.size(0)
                val_loss /= len(val_loader.dataset)
                val_loss_list.append(val_loss)
                if val_loss < best_valid_loss:
                    best_valid_loss=val_loss
                    best_weights = copy.deepcopy(self.model.state_dict())
            print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, self.epochs, train_loss, val_loss))
        #self.plot_loss(train_loss_list,val_loss_list,self.epochs)
        self.saver(best_weights)
        return best_valid_loss
    
    def tester(self,X,Y):
        device=self.init_device()
        name=''.join(f"{key}{val}" for key, val in self.model.__repr__().items())
        path="./models/"+name
        self.model.load_state_dict(torch.load(path))
        test_loader=self.dataloader(X,Y)
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                output = self.model(x)
                loss = self.criterion(output, y)
                test_loss += loss.item()*x.size(0)
            test_loss /= len(test_loader.dataset)
            print('Test Loss: {:.6f}'.format(test_loss))
        return test_loss