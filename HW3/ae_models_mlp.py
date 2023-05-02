import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def train_mlp(Xtrain, Ytrain, model, optimizer):
    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model(Xtrain.float())
        loss = nn.MSELoss()(y_pred.view(-1), Ytrain.view(-1))
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")
    
def grid_search_mlp(Xtr, Ytr, xval, yval, learning_rates, hidden_sizes, batch_sizes):
    best_loss = float('inf')
    best_hyperparams = {}
    # Loop over all combinations of hyperparameters
    for lr in learning_rates:
        for hidden_size in hidden_sizes:
            for batch_size in batch_sizes:
                # Define model and optimizer
                model = MLP(input_dim=7, hidden_dim=hidden_size, output_dim=1)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                
                # Train model
                num_epochs = 1000
                for epoch in range(num_epochs):
                    for i in range(0, len(Xtr), batch_size):
                        optimizer.zero_grad()
                        y_pred = model(Xtr[i:i+batch_size].float())
                        loss = nn.MSELoss()(y_pred.view(-1), Ytr[i:i+batch_size].view(-1))
                        loss.backward()
                        optimizer.step()
                    
                    if epoch % 100 == 0:
                        print(f"Epoch {epoch}: Loss = {loss.item()}")
                
                # Evaluate model on val set
                y_pred = model(xval.float())
                loss = nn.MSELoss()(y_pred.view(-1), yval.view(-1))
                print(f"lr={lr}, hidden_size={hidden_size}, batch_size={batch_size}, test_loss={loss.item()}")
                
                # Check if this is the best set of hyperparameters
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_hyperparams = {'lr': lr, 'hidden_size': hidden_size, 'batch_size': batch_size}

    print(f"Best hyperparameters: {best_hyperparams}")




