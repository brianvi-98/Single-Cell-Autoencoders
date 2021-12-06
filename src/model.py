from scipy import sparse
import sys
import os
import json


import anndata as ad
import numpy as np
import pandas as pd
import logging
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import random

import anndata as ad
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE

import plotly.io as plt_io
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks')
from sklearn.metrics import mean_squared_error
from tensorflow import keras

from keras.models import Model
from keras.layers import Dense, Input
from keras.regularizers import l1
from tensorflow.keras.optimizers import Adam 
import tensorflow as tf
tf.config.run_functions_eagerly(True)

import torch

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import euclidean_distances
from scipy import sparse

class AE_adt(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        #self.encoder_hidden_layer5000 = nn.Linear(in_features=kwargs["input_shape"], out_features=5000)
        
        self.encoder_hidden_layer1 = nn.Linear(in_features=kwargs["input_shape"], out_features=80)
        self.encoder_hidden_layer2 = nn.Linear(in_features=80, out_features=40)
        self.encoder_output_layer = nn.Linear(in_features=40, out_features=2)
        
        self.decoder_hidden_layer1 = nn.Linear(in_features=2, out_features=40)
        #self.decoder_hidden_layer5000 = nn.Linear(in_features=250, out_features=5000)
        self.decoder_hidden_layer2 = nn.Linear(in_features=40, out_features=80)
        self.decoder_output_layer = nn.Linear(in_features=80, out_features=kwargs["input_shape"])

    def forward(self, features):
        #activation = self.encoder_hidden_layer5000(features)
        #activation = torch.relu(activation)
        
        activation = self.encoder_hidden_layer1(features)
        activation = torch.relu(activation)
        activation = self.encoder_hidden_layer2(activation)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
#         code = torch.relu(code)
        activation = self.decoder_hidden_layer1(code)
        activation = torch.relu(activation)
        activation = self.decoder_hidden_layer2(activation)
        activation = torch.relu(activation)
        
        #activation = self.decoder_hidden_layer5000(activation)
        #activation = torch.relu(activation)
        
        activation = self.decoder_output_layer(activation)
#         reconstructed = torch.relu(activation)
        return [code, activation]
class AE_gex(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer256 = nn.Linear(in_features=kwargs["input_shape"], out_features=1000)
        self.normalize = nn.LayerNorm(1000)
        self.encoder_hidden_layer128 = nn.Linear(in_features=1000, out_features=128)
        self.encoder_hidden_layer64 = nn.Linear(in_features=128, out_features=64)
        self.encoder_output_layer = nn.Linear(in_features=64, out_features=2)
        
        self.decoder_hidden_layer = nn.Linear(in_features=2, out_features=64)
        self.decoder_hidden_layer64d = nn.Linear(in_features=64, out_features=128)
        self.decoder_hidden_layer128d = nn.Linear(in_features=128, out_features=1000)
        self.decoder_output_layer = nn.Linear(in_features=1000, out_features=kwargs["input_shape"])

    def forward(self, features):
        activation = self.encoder_hidden_layer256(features)
        activation = torch.relu(activation)
        activation = self.normalize(activation)
        activation = self.encoder_hidden_layer128(activation)
        activation = torch.relu(activation)
        activation = self.encoder_hidden_layer64(activation)
        code = self.encoder_output_layer(activation)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_hidden_layer64d(activation)
        activation = torch.relu(activation)
        activation = self.decoder_hidden_layer128d(activation)
        activation = torch.relu(activation)    
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return [code,activation]
    
def my_loss_test(code,curbatch):
    torch_dist_matrix = torch.cdist(cur_batch,cur_batch)
    D = (torch.cdist(code,code))
    return (1/code.shape[0])*torch.sum(torch.absolute(torch_dist_matrix-D))    

def predict_mod(mod,test_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prediction = mod(test_data.to(device))[1]
    return nn.MSELoss()(test_data.to(device),prediction)

def train_model(input_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adt_torch_data = input_data.to(device)
    model_adt = AE(input_shape=134).to(device)
    optimizer_adt = optim.Adam(model_adt.parameters(), lr=1e-3)
    criterion_mse = nn.MSELoss()
    for epoch in range(5):
        loss = 0
        for i in range(0,200,100):
            cur_batch = adt_torch_data[i:i+100]
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer_adt.zero_grad()

            # compute reconstructions
            outputs = model_adt(cur_batch)[1]
            code_output = model_adt(cur_batch)[0]

            train_loss = criterion_mse(outputs, cur_batch)
            #train_loss = criterion(code_output,cur_batch)+train_loss_mse

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer_adt.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / 200
 
    # display the epoch training loss
        print('Training...Epoch: '+ str(epoch+1))
        #print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, 5, loss))
    return model_adt
    
def build_model(adt_torch_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adt_torch_data = adt_torch_data.to(device)
    model_adt = AE(input_shape=134).to(device)
    optimizer_adt = optim.Adam(model_adt.parameters(), lr=1e-3)
    criterion_mse = nn.MSELoss()
    for epoch in range(5):
        loss = 0
        for i in range(0,200,100):
            cur_batch = adt_torch_data[i:i+100]
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer_adt.zero_grad()

            # compute reconstructions
            outputs = model_adt(cur_batch)[1]
            code_output = model_adt(cur_batch)[0]

            train_loss = criterion_mse(outputs, cur_batch)
            #train_loss = criterion(code_output,cur_batch)+train_loss_mse

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer_adt.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / 200
 
    # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, 5, loss))
   
    

