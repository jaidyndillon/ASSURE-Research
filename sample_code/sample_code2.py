import torch 
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torchmetrics import PearsonCorrCoef
import numpy as np

#Define Hyper-parameters (fixed variables)
input_size = 10000
hidden_size = 500 
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
    
    geno0 = np.loadtxt("/home/jfdillon/thinclient_drives/Download/python_files/txt.raw", dtype = "float32", skiprows = 1, usecols = range(6, 10006))
    #uses np to load the file
    #variables provided are: data type, skip rows (skips the rows you specify), usecols (selects specific columns to retrieve)
    phen0 = np.loadtxt("/home/jfdillon/thinclient_drives/Download/python_files/phen.csv", dtype = "float32", skiprows = 1, delimiter = ",", usecols = range(1, 4))
    #delimiter defines what will be used to separate values (in this case it's a comma)
    geno0 = StandardScaler().fit_transform(geno0)
    phen0 = StandardScaler().fit_transform(phen0)
    #Standard Scaler removes the mean and scales each feature/variable to unit variance
    #it can be influenced by outliers (makes a 0 mean and unit standard-deviation)
    
    train_dataset = GenDataset(geno0[0:2000,], phen0[0:2000, 0 ])
    test_dataset = GenDataset(geno0[2000:,], phen0[2000:, 0])
    test_abs = int(len(train_dataset)*0.95)
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [test_abs, len(train_dataset) - test_abs])
    #validation subset takes a subset of training set to run 
    #tests if model is just memorizing or learning
    
    
    #Data Loader
    train_loader = torch.utils.data.DataLoader(dataset = train_subset, batch_size = batch_size, shuffle = True)
    val_loader = torch.utils.data.DataLoader(dataset = val_subset, batch_size = batch_size, shuffle = False) 
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False) 
    
    #FC NN 
    class NeuralNet(nn.Module): 
        def __init__(self, input_size, hidden_size, num_traits): 
            super(NeuralNet, self).__init__() 
            self.fc1 = nn.Linear(input_size, num_traits) #puts the inputs through one linear layer that outputs the size of num_traits
        
        def forward(self, x): 
            out = self.fc1(x) 
            return out
        
    model = NeuralNet(input_size, hidden_size, num_traits).to(device)
    
    #Loss and Optimizer 
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 10) 
    #weight decay helps ensure that model isn't overfitting 
    #if too low, it can result in a highly flexible model that can result in overfitting (too general model)
    #if too high, it can result in an inflexible model that can result in underfitting (more specific model)

    pearson = PearsonCorrCoef()

    #Train the model 
    total_step = len(train_loader) 
    for epoch in range(num_epochs): 
        for i, (geno, phen) in enumerate(train_loader):
            geno = geno.to(device)
            phen = phen.to(device)
            #move them to certain device 
            
            outputs = model(geno) 

            #ensures shapes are the same
            #print("outputs:", outputs.shape)
            #print("outputs after flatten:", torch.flatten(outputs).shape)
            #print("targets:", phen.shape)

            loss = criterion(torch.flatten(outputs), phen)

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step() 
            if epoch % 10 == 0 and (i+1) % 1 == 0: 
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
        with torch.no_grad(): #runs the validation set through without calculating gradients
            for geno, phen in val_loader: 
                geno = geno.to(device)
                phen = phen.to(device)

                outputs = model(geno)
                acc = pearson(torch.flatten(outputs), phen)
                if epoch % 10 == 0: 
                    print("Accuracy of the network on the val data: {:.4f}".format(acc))
    
    with torch.no_grad(): #runs test set through without calculating gradients
        for geno, phen in test_loader:
            geno = geno.to(device)
            phen = phen.to(device)

            outputs = model(geno)
            acc = pearson(torch.flatten(outputs), phen)
            print("Accuracry of the network on the test data: {:.4f}".format(acc))
