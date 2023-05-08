import numpy as np
import torch
import torch.nn as nn
import  matplotlib.pyplot as plt
import ae


data       = np.load("../data/ecg_data.npz")
Xtr        = data["Xtr"]       #Clean train
Xtr_noise  = data["Xtr_noise"] #Noisy train
Xte_noise  = data["Xte_noise"] #Noisy test

params = np.load("../data/ecg_params.npz")
W = torch.FloatTensor(params["W"]) #Decoder parameters
V = torch.FloatTensor(params["V"]) #Encoder parameters