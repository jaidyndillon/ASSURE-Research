import time 
import torch
import numpy as np 
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()

input_size = 10000
n_traits = 1 
num_epochs = 400 
batch_size = 100
learning_rate = 1e-4

start = time.time()

class GenDataset (torch.utils.data.Dataset): 
    def __init__(self, X, y): 
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)
    def __len__(self): 
        return len(self.X) #
    def __getitem__(self, i): 
        return self.X[i], self.y[i]  

columns_to_load = range(6, input_size + 6, 5) #selects every 5 columns b/c GPU memory is exceeded
#if GPU memory is still exceeded with this, you can increase from 5 to 10, etc.; just make sure to change the input_size and nhead appropriately (detailed below)

geno0 = np.loadtxt("/home/jfdillon/thinclient_drives/Download/python_files/txt.raw", 
                   dtype = "float32", skiprows = 1, usecols = columns_to_load)
phen0 = np.loadtxt("/home/jfdillon/thinclient_drives/Download/python_files/phen.csv", 
                   dtype = "float32", skiprows = 1, delimiter = ",", usecols = range(1, 2))

#phen0 = np.reshape(phen0, (-1, 1)) if using range 1,4
phen0 = np.reshape(phen0, (phen0.shape[0], 1))

geno0 = StandardScaler().fit_transform(geno0)
phen0 = StandardScaler().fit_transform(phen0)

print(geno0.shape)
print(phen0.shape)

end = time.time() 
print(end-start)
start = end

#Create Datasets
train_dataset = GenDataset(geno0[0:2000,], phen0[0:2000, 0])
test_dataset = GenDataset(geno0[2000:,], phen0[2000:, 0])

train_abs = int(len(train_dataset) * 0.9)

train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_abs, len(train_dataset) - train_abs])


#Create Dataloaders
train_loader = torch.utils.data.DataLoader(dataset = train_subset, 
                                           batch_size = batch_size, 
                                           shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = val_subset,
                                          batch_size = batch_size, 
                                          shuffle = False) 
val_loader = torch.utils.data.DataLoader(dataset = test_dataset, 
                                         batch_size = batch_size, 
                                         shuffle = False)

class TransformerModel(nn.Module): 
    def __init__(self, input_dim, output_dim, num_layers, nhead, hidden_dim, dropout): 
        super(TransformerModel, self).__init__() 
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = input_dim, nhead = nhead, dim_feedforward = hidden_dim, dropout = dropout, batch_first = True) 

        #d_model = input_dim(number of expected features in the input)
        #nhead = number of heads in multiattention model
        #dim_feedforward = hidden_dim (dimension of the feedforward network)
        #batch_first = input and output tensors are (batch, seq, feature)

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = num_layers)
        self.decoder = nn.Linear(input_dim, output_dim)
        
    def forward(self, x): 
        x = self.transformer_encoder(x) 
        x = self.decoder(x) 
        return x

input_dim = 2000 #would be input_size / step value in columns_to_load
output_dim = 1 
num_layers = 4
nhead = 8 #change this according to input_dim (ensure input_dim is divisible by it)
hidden_dim = 516
dropout = 0.5

model  = TransformerModel(input_dim, output_dim, num_layers, nhead, hidden_dim, dropout).to(device)

#Loss and Optimizer 
criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 10)

#Model Training
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    total_step = len(train_loader)
    best_val_acc = -1.0 
    patience = 10 
    counter = 0 

    for epoch in range(num_epochs): 
        for i, (geno, phen) in enumerate(train_loader): 
            geno = geno.to(device)
            phen = phen.to(device) 
        
            #Forward 
            outputs = model(geno)
            loss = criterion(outputs.squeeze(), phen) 
        
            #Backward 
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
        
            if epoch % 10 == 0 and (i+1) % 1 == 0: 
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        
        with torch.no_grad(): 
            val_outputs_list = [] 
            val_phen_list = [] 
            for geno, phen in val_loader: 
                geno = geno.to(device) 
                phen = phen.to(device) 
            
                outputs = model(geno)
            
                #Add outputs and targets to different lists to view
                val_outputs_list.append(outputs.squeeze())
                val_phen_list.append(phen)
        
            val_outputs_tensor = torch.cat(val_outputs_list)
            val_phen_tensor = torch.cat(val_phen_list)
        
            cat = torch.stack((val_outputs_tensor, val_phen_tensor), dim = 0)
        
            val_acc = torch.corrcoef(cat)[0,1]
        
            if epoch % 10 == 0:
                print('Accuracy of the network on the val data: {:.4f}'.format(val_acc))
        
            #Early stopping
           
            if val_acc > best_val_acc: 
                best_val_acc = val_acc 
                counter = 0 
            else: 
                counter += 1 
                if counter >= patience: 
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
#Model testing (don't need to compute gradients for memory efficiency)
def test(model, test_loader):
    with torch.no_grad(): 
        outputs_list = [] 
        phen_list = [] 
        for geno, phen in test_loader: 
            geno = geno.to(device)
            phen = phen.to(device)
        
            outputs = model(geno) 
            outputs_list.append(outputs.squeeze())
            phen_list.append(phen) 
        
        outputs_tensor = torch.cat(outputs_list) 
        phen_tensor = torch.cat(phen_list)
    
        cat = torch.stack((outputs_tensor, phen_tensor), dim = 0)
        acc = torch.corrcoef(cat)[0,1]

    # Print the final learning rate
    print(f"Final learning rate: {scheduler.optimizer.param_groups[0]['lr']:.6f}")

    end = time.time()
    print(end-start)
    
if __name__ == "__main__": 
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs)
    test(model, test_loader)
