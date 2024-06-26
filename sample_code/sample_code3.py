import torch 
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torchmetrics import PearsonCorrCoef
import numpy as np

input_size = 10000
hidden_size = 2000
num_traits = 1 
num_epochs = 250 
batch_size = 1000
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
    
    geno0 = np.loadtxt("/Users/jaidyndillon/Downloads/txt.raw", dtype = "float32", skiprows = 1, usecols = range(6, 10006))
    #uses np to load the file
    #variables provided are: data type, skip rows (skips the rows you specify), usecols (selects specific columns to retrieve)
    phen0 = np.loadtxt("/Users/jaidyndillon/Downloads/phen.csv", dtype = "float32", skiprows = 1, delimiter = ",", usecols = range(1, 4))
    #delimiter defines what will be used to separate values (in this case it's a comma)
    geno0 = StandardScaler().fit_transform(geno0)
    phen0 = StandardScaler().fit_transform(phen0)
    #Standard Scaler removes the mean and scales each feature/variable to unit variance
    #it can be influenced by outliers (makes a 0 mean and unit standard-deviation)
    
    train_dataset = GenDataset(geno0[0:2000,], phen0[0:2000, 0])
    test_dataset = GenDataset(geno0[2000:,], phen0[2000:, 0])
    
    #Data Loader
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)
    
    
    #FC NN Layers
    class NeuralNet(nn.Module): 
        def __init__(self, input_size, hidden_size, num_traits): 
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.tanh = nn.Tanh() #applies hyperbolic tangent
            self.do = nn.Dropout(0.1) #creates a dropout layer with a frequency of 0.1
            self.fc2 =  nn.Linear(hidden_size, hidden_size) 
            self.fc3 = nn.Linear(hidden_size, num_traits)
            
        def forward(self, x): 
            out = self.fc1(x)
            out = self.relu(out)
            #out = self.do(out) #would pass out through this dropout layer
            #out = self.fc2(out) #would pass out through second layer, is skipped because trying to see how accuracy and loss is when only ran through 2 layers
            #out = self.relu(out)
            #out = self.do(out) 
            out = self.fc3(out) 
            return out
    
    model = NeuralNet(input_size, hidden_size, num_traits)
    
    #Loss and Optimizer 
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1)
    
    #Training Model
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
            if epoch % 10 == 0 and (i+1) % 1 == 0: 
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, i+1, total_step, loss.item()))
    
    pearson = PearsonCorrCoef() 
    with torch.no_grad(): 
        for geno, phen in test_loader:
            geno = geno.to(device)
            phen = phen.to(device)
            #move them to certain device 
 
            outputs = model(geno) 
            acc = pearson(torch.flatten(outputs), phen)
            print("Accuracy of the network on the test data: {:.4f}".format(acc))
        