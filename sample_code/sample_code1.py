import torch
import torch.nn as nn 
from sklearn.preprocessing import StandardScaler
from torchmetrics import PearsonCorrCoef
import numpy as np

#Check Devide Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Define Hyper-parameters (fixed variables)
input_size = 10000
#only 10000 inputs
hidden_size = 500
#size of input for hidden layers (act as a memory for what's already been seen in sequence)
num_traits = 1
#looking at only 1 trait 
num_epochs = 10
#runs through the model 10 times
batch_size = 1000
#batches of 1000 inputs (since input size is 10000 can only have 10 batches)
learning_rate = 0.1 
#size of steps that the model takes to a better prediction

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
        def size(self): 
            return self.X.size(), self.y.size()
    
    geno0 = np.loadtxt("/home/jfdillon/thinclient_drives/Download/python_files/txt.raw", dtype = "float32", skiprows = 1, usecols = range(6, 10006))
    #uses np to load the file\
    #variables provided are: data type, skip rows (skips the rows you specify), usecols (selects specific columns to retrieve)
    phen0 = np.loadtxt("/home/jfdillon/thinclient_drives/Download/python_files/phen.csv", dtype = "float32", skiprows = 1, delimiter = ",", usecols = range(1, 4))
    #delimiter defines what will be used to separate values (in this case it's a comma)
    geno0 = StandardScaler().fit_transform(geno0)
    phen0 = StandardScaler().fit_transform(phen0)
    #Standard Scaler removes the mean and scales each feature/variable to unit variance
    #it can be influenced by outliers (makes a 0 mean and unit standard-deviation)
    
    train_dataset = GenDataset(geno0[0:2000,], phen0[0:2000,1])
    test_dataset = GenDataset(geno0[2000:,], phen0[2000:,1]) #last number for phen0 tells the increment
    
    print(train_dataset.size())
    print(test_dataset.size())
    
    #Data Loader
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    #train_dataset is loaded in batches the size of the batch_size and is randomized
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)
    #test_dataset is loaded in batches the size of the batch_size and isn't randomized
    
    #Fully connected (FC) NN
    class NeuralNet(nn.Module): 
        def __init__(self, input_size, hidden_size, num_traits): 
            super(NeuralNet, self).__init__() #must be done to initiate nn
            self.fc1 = nn.Linear(input_size, hidden_size) 
            #linear layer that takes the input_size amount of inputs and compiles it into hidden_size amount
            self.relu = nn.ReLU() #linearly adds to the deep learning
            #self.do1 = nn.Dropout(0.5)
                #is an option if you want to include a dropout layer that would set random parts of input to 0 at a frequency of 0.5
            self.fc2 = nn.Linear(hidden_size, num_traits)
            #linear layer that takes the hidden_size amount of inputs and compiles it into num_traits amount
            
        def forward(self, x): 
            out = self.fc1(x) #assigns self.fc1 of input to out
            out = self.relu(out) #compiles what is in self to out
            #out = self.do1(out) 
            #passes out through the dropout layer (in this case, it won't because of the #)
            out = self.fc2(out) #passes input through self.fc2
            return out 
    
    model = NeuralNet(input_size, hidden_size, num_traits).to(device) 
    #calls on nn function
    
    #Loss and Optimizer
    criterion = nn.MSELoss() #defines which loss calculation to use
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    #Adam = Adaptive Moment Estimation
    #adaptive way to adjust learning rates to ensure optimized learning 
    
    #can create two functions for train and test     

    #Train the Module
    total_step = len(train_loader) 
    for epoch in range(num_epochs): 
        for i, (geno, phen) in enumerate(train_loader): #tracks number of iterations in a loop for geno and pheno
            geno = geno.to(device)
            phen = phen.to(device)
            #move them to certain device 
            
            #Forward pass
            outputs = model(geno) 
            
            #used to ensure shapes are the same
            #print("Outputs:", outputs.shape) 
            #print("Outputs flatten:", torch.flatten(outputs).shape)
            #print("Targets:", phen.shape) 

            loss = criterion(torch.flatten(outputs), phen)
            #calculates loss between output after being flattened and the pheno
            
            #Backward pass 
            optimizer.zero_grad() #set gradients to 0
            loss.backward() #compute gradient from loss for each input and puts them into tensor's grad attribute 
            optimizer.step() #initates optimizer
            if (i+1) % 1 == 0: 
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch +1, num_epochs, i+1, total_step, loss.item()))
    
    pearson = PearsonCorrCoef().to(device) 
    #measures strength and direction of linear relationship between two variables 
    
    #Test the Model (don't need to compute gradients for memory efficiency)
    with torch.no_grad(): #turns off autograd
        for geno, phen in test_loader: 
            geno = geno.to(device)
            phen = phen.to(device) 
            outputs = model(geno.cuda())
            print("Outputs:", outputs.shape)
            print("Outputs flatten:", torch.flatten(outputs).shape)
            print("Targets:", phen.shape)
            acc = pearson(torch.flatten(outputs), phen) #strength and direction btwn flattened outputs and phenotype
            print("Accuracy of the network on the test data: {:.4f}".format(acc))
    
    #Save the model checkpoint
    #torch.save(model.state_dict(), "gs.model.ckpt")
    
