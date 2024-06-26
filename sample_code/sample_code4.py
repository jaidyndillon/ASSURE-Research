#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:02:29 2024

@author: jaidyndillon
Using RNN (Recurrent NN)
RNN are used for sequential data 
"""

import torch 
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torchmetrics import PearsonCorrCoef
import numpy as np

input_size = 10000 
seg_size = 1 
input_size //= seg_size #floor divides the input_size and assigns it to input_size 
#floor divide: divides input_size by seg_size and assigns the interger of the quotient to input_size

num_layers = 1 
hidden_size = 1000 
num_traits = 1 
num_epochs = 400
batch_size = 2000
learning_rate = 1e-4 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#The following sample codes include the same GenDataset class and geno0 and phen0 definitions 

if __name__ == "__main__": 
    class GenDataset (torch.utils.data.Dataset): #GenDataset accesses and processes single instances from dataset
        def __init__(self, X, y): #defines this layer using self, x, and y
            if not torch.is_tensor(X) and not torch.is_tensor(y):
                self.X = torch.from_numpy(X) #creates a tensor using same shape and data as numpy array
                self.y = torch.from_numpy(y)
        def __len__(self): #defines __len__ using self
            return len(self.X) #returns the number of elements in X tensor
        def __getitem__(self, i): #defines _-getitem__ using self and i
            return self.X[i], self.y[i] #returns the item in tensor X and tensor y
    
    geno0 = np.loadtxt("/Users/jaidyndillon/Downloads/python_files/txt.raw", dtype = "float32", skiprows = 1, usecols = range(6, 10006))
    #uses np to load the file
    #variables provided are: data type, skip rows (skips the rows you specify), usecols (selects specific columns to retrieve)
    phen0 = np.loadtxt("/Users/jaidyndillon/Downloads/python_files/phen.csv", dtype = "float32", skiprows = 1, delimiter = ",", usecols = range(1, 4))
    #delimiter defines what will be used to separate values (in this case it's a comma)
    geno0 = StandardScaler().fit_transform(geno0)
    phen0 = StandardScaler().fit_transform(phen0)
    #Standard Scaler removes the mean and scales each feature/variable to unit variance
    #it can be influenced by outliers (makes a 0 mean and unit standard-deviation)
    
    geno1 = np.reshape(geno0, (3000, seg_size, -1)) #changing the shape of the array 
    print(geno1)
    
    train_dataset = GenDataset(geno1[0:2000,], phen0[0:2000, 1])
    test_dataset = GenDataset(geno1[2000:,], phen0[2000:, 1])
    
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)
    
    #FC NN 
    class NeuralNet(nn.Module): 
        def __init__(self, input_size, hidden_size, num_layers, num_traits): 
            super(NeuralNet, self).__init__() 
            self.num_layers = num_layers #establishing number of layers
            self.hidden_size = hidden_size #establishing hidden_size
            #RNN layer
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity = "relu", batch_first = True) 
            #batch_first being true means it uses the second dimension(hidden_size) as opposed to the first
            #uses ReLU as the nonlinearity method
            #FC layer
            self.fc = nn.Linear(hidden_size, num_traits) 
            #takes the output from RNN in the hidden_size and breaks it down to num_traits size
            
        def forward(self, x): 
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            #creates the number of layers of tensor of 0's that has the size of x rows and hidden_size amount of columns
            out, hn = self.rnn(x, h0) #passes x and h0 through rnn 
            out = self.fc(out[:,-1,:])
            return out
    
    model = NeuralNet(input_size, hidden_size, num_layers, num_traits)
    
    #Loss and Optimizer 
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 0.50)
    
    total_step = len(train_loader) 
    for epoch in range(num_epochs): 
        for i, (geno, phen) in enumerate(train_loader): 
            geno = geno.to(device)
            phen = phen.to(device)
            #move them to certain device 
            
            outputs = model(geno) 
            loss = criterion(torch.flatten(outputs), phen)

            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
            if epoch % 10 == 0 and (i+1) %1 ==0: 
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                
    pearson = PearsonCorrCoef() 
    with torch.no_grad(): 
        test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)
        for geno, phen in test_loader: 
            geno = geno.to(device)
            phen = phen.to(device) 

            outputs = model(geno) 
            acc = pearson(torch.flatten(outputs), phen)
            print("Accuracy of the network on the test data: {:.4f}".format(acc))
        
    
    
    
