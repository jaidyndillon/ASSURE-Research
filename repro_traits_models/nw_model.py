import time
import torch 
import numpy as np 
import torch.nn as nn 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.stats import uniform
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score, explained_variance_score

start = time.time()

#Set random seed for reproducibility for RandomizedSearchCV
#np.random.seed(1) 
#torch.manual_seed(1) 
#torch.cuda.manual_seed_all(1)

#Check Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device) 
torch.cuda.empty_cache() #empties memory cache

#TF Hyperparameters
input_size = 30980 
num_traits = 1 
num_epochs = 200 
batch_size = 100
learning_rate = 0.001

total = 3290
train_percent = int(total * 0.8)

#Modify Data Inputs 
class GenDataset(torch.utils.data.Dataset): 
    def __init__(self, X, y): 
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)
    def __len__(self): 
        return len(self.X) #
    def __getitem__(self, i): 
        return self.X[i], self.y[i]  

use_columns = range(6, input_size + 6, 5)

nw_geno = np.loadtxt("C:/Users/Jaidyn/Documents/ASSURE/python/project/repro_traits_model/NW/spg_NW.raw", dtype = "float32",
                     skiprows = 1, usecols = use_columns)
nw_pheno = np.loadtxt("C:/Users/Jaidyn/Documents/ASSURE/python/project/repro_traits_model/NW/NW_pheno.csv", dtype = "float32", 
                      skiprows = 1, delimiter = ",", usecols = range(1,2))
nw_pheno = np.reshape(nw_pheno, (nw_pheno.shape[0], 1))

nw_geno = StandardScaler().fit_transform(nw_geno) 
nw_pheno = StandardScaler().fit_transform(nw_pheno)

#Check Geno and Pheno Shapes
print("nw geno:", nw_geno.shape)
print("nw pheno:", nw_pheno.shape)

#Check how long data loading took
end = time.time() 
print(end - start)
start = end

#Datasets 
nw_train_dataset = GenDataset(nw_geno[0:train_percent,], nw_pheno[0:train_percent, 0])
nw_test_dataset = GenDataset(nw_geno[train_percent:,], nw_pheno[train_percent:, 0])

#Create Test Abs
nw_test_abs = int(len(nw_train_dataset) * 0.8)
nw_train_subset, nw_val_subset = torch.utils.data.random_split(nw_train_dataset, [nw_test_abs, len(nw_train_dataset) - nw_test_abs])

#Check Division of Datasets
#print("Train dataset:", len(nba_train_dataset))
#print("Test dataset:", len(nba_test_dataset))
#print("Train subset:", len(nba_train_subset))
#print("Val subset: ", len(nba_val_subset))

#DataLoader
nw_train_loader = torch.utils.data.DataLoader(dataset = nw_train_subset, 
                                              batch_size = batch_size, 
                                              shuffle = True) 
nw_val_loader = torch.utils.data.DataLoader(dataset = nw_val_subset, 
                                         batch_size = batch_size,
                                         shuffle = False)
nw_test_loader = torch.utils.data.DataLoader(dataset = nw_test_dataset, 
                                          batch_size = batch_size, 
                                          shuffle = False)

#Define FlashAttentionLayer
class FlashAttentionLayer(nn.Module): 
    def __init__(self, d_model, nheads, dropout): 
        super(FlashAttentionLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nheads, dropout = dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        #Initialize weights
        nn.init.xavier_uniform_(self.self_attn.in_proj_weight) 
        nn.init.xavier_uniform_(self.self_attn.out_proj.weight)
        
    def forward(self, src, src_mask = None): 
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout(src2) 
        src  = self.layer_norm(src) 
        return src

#Transformer Model
class TransformerModel(nn.Module): 
    def __init__(self, input_dim, output_dim, nlayer, nhead, hidden_dim, dropout): 
        super(TransformerModel, self).__init__() 
        self.encoder_layers = nn.ModuleList([FlashAttentionLayer(input_dim, nhead, dropout) for _ in range(nlayer)])
        self.decoder = nn.Linear(input_dim, output_dim)
        self.init_weights() 
        
    def init_weights(self): 
        #Initialize weights for decoder layer
        self.decoder.weight = nn.init.kaiming_uniform_(self.decoder.weight, mode = "fan_in", nonlinearity = "relu")
        self.decoder.bias.data.fill_(0)

    def forward(self, x): 
        for layer in self.encoder_layers: 
            x = layer(x)
        
        x = self.decoder(x) #gives [batch_size, seq_length, output_dim]
        return x
    

#Model Hyperparameters
input_dim = 6196
output_dim = 1 
nlayer = 6
nhead = 2
hidden_dim = 768
dropout = 0.4
weight_decay = 6

model = TransformerModel(input_dim, output_dim, nlayer, nhead, hidden_dim, dropout).to(device)

#Loss and Optimizer 
criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

#LR Scheduler 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "max",factor = 0.1, patience = 5)

#Model Training 
def train(model, nw_train_loader, nw_val_loader, criterion, optimizer, num_epochs): 
    best_val_acc = -1.0 
    patience = 10
    counter = 0 
    all_acc = [] 
    all_r2 = [] 
    fold = 1
    
    for epoch in range(num_epochs): 
            model.train() 
            train_loss = 0.0 
            
            for geno, pheno in nw_train_loader: 
                geno = geno.to(device)
                pheno = pheno.to(device) 
                
                #Forward 
                outputs = model(geno)
                loss = criterion(outputs.squeeze(), pheno.squeeze())
                
                #Backward
                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if epoch % 10 == 0 and (1+1) % 1 == 0: 
                    print("Epoch [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, loss.item()))
                    
            #Validation 
            model.eval() 
            val_outputs_list = [] 
            val_pheno_list= [] 
            
            with torch.no_grad(): 
                for geno, pheno in nw_val_loader: 
                    geno = geno.to(device) 
                    pheno = pheno.to(device)
                    
                    outputs = model(geno)
                    
                    #Add outputs and targets to different lists 
                    val_outputs_list.append(outputs.squeeze())
                    val_pheno_list.append(pheno.squeeze())
                    
                val_outputs_tensor = torch.cat(val_outputs_list).cpu()
                val_pheno_tensor = torch.cat(val_pheno_list).cpu()
                
                cat = torch.stack((val_outputs_tensor, val_pheno_tensor), dim = 0)
                
                val_acc = torch.corrcoef(cat)[0,1]
                #Correlation Coeficient meausures how well relationship between predicted and true values matches linear relationship 
                
                #Calculate R2 
                #R2 measures how well the model approximates the real data points (higher indicates a better fit) 
                r2 = r2_score(val_pheno_tensor.cpu().numpy(), val_outputs_tensor.cpu().numpy())
                print(f"Epoch [{epoch + 1}/ {num_epochs}], Validation Accuracy: {val_acc:.4f}, Validation R2: {r2:.4f}")
                
                if val_acc > best_val_acc: 
                    best_val_acc = val_acc
                    counter = 0
                else: 
                    counter += 1 
                    if counter >= patience: 
                        print(f"Early stopping at epoch {epoch+1}")
                        return best_val_acc
            all_acc.append(best_val_acc)
            all_r2.append(r2)
            fold += 1
        
    mean_acc = np.mean(all_acc)
    mean_r2 = np.mean(all_r2)
    print(f"Mean Cross-validation corrcoef: {mean_acc:.4f}, Mean R2: {mean_r2:.4f}")
    return mean_acc, mean_r2

#Model Testing 
def test(model, nw_test_loader): 
    with torch.no_grad(): 
        outputs_list = [] 
        pheno_list = [] 
        for geno, pheno in nw_test_loader:
            geno = geno.to(device)
            pheno = pheno.to(device)
            
            outputs = model(geno)
           
            #Check Output and Pheno Sizes
            #print("Outputs:", outputs.shape)
            #print("Oututs squeezed: ", outputs.squeeze().shape)
            #print("Pheno: ", pheno.shape)
            #print("Pheno squeezed: ", pheno.squeeze().shape)
            
            outputs_list.append(outputs.squeeze())
            pheno_list.append(pheno.squeeze())

        if outputs_list and pheno_list: 
            outputs_tensor = torch.cat(outputs_list).cpu()
            pheno_tensor = torch.cat(pheno_list).cpu()
        
            cat = torch.stack((outputs_tensor, pheno_tensor), dim = 0)
            acc = torch.corrcoef(cat)[0, 1]
            
            #Calculate R2 
            r2 = r2_score(pheno_tensor.cpu().numpy(), outputs_tensor.cpu().numpy())
        else: 
            print("Validation lists are empty or incompatible")
            
    print("Accuracy of Model: {}".format(acc))
    print("R2 score: {:.4f}".format(r2))
    
    end = time.time() 
    print(end - start)
    
    return acc.item(), r2
    #print(f"Final learning rate: {scheduler.optimizer.param_groups[0] ['lr']:.6f}")

#Train with CV 
def train_with_cv(model, criterion, optimizer, X, y, num_epochs, batch_size, n_splits = 5): 
    
    random_state = 42 #Ser random state for reproducibility 
    shuffle_indices = np.random.permutation(len(nw_geno))
    nw_geno_shuffled = nw_geno[shuffle_indices]
    nw_pheno_shuffled = nw_pheno[shuffle_indices] 
    
    kf = KFold(n_splits = n_splits, shuffle = True, random_state = random_state) 
    for fold_idx, (train_index, test_index) in enumerate(kf.split(nw_geno_shuffled)): 
        print(f"Fold [{fold_idx + 1}/{n_splits}]")
        
        X_train, X_test, y_train, y_test = train_test_split(nw_geno_shuffled, nw_pheno_shuffled, test_size=0.2, random_state=42)
        
        #print("Train size: ", X_train.shape)
        #print("Test size: ", X_test.shape)
        
        train_subset_size = int(0.8* len(X_train))
        
        X_train_subset = X_train[:train_subset_size]
        y_train_subset = y_train[:train_subset_size]
        
        X_val = X_train[train_subset_size:]
        y_val = y_train[train_subset_size:]
        
        #print("Train subset size: ", X_train_subset.shape)
        #print("Val size: ", X_val.shape)
        
        cv_train_dataset = GenDataset(X_train_subset, y_train_subset)
        cv_val_dataset = GenDataset(X_val, y_val)
        cv_test_dataset = GenDataset(X_test, y_test)
        
        cv_train_loader = torch.utils.data.DataLoader(dataset = cv_train_dataset, batch_size = batch_size, shuffle = True) 
        cv_val_loader = torch.utils.data.DataLoader(dataset = cv_val_dataset, batch_size = batch_size, shuffle = False)
        cv_test_loader = torch.utils.data.DataLoader(dataset = cv_test_dataset, batch_size = batch_size, shuffle = False) 
        
        #print("Train loader size: ", len(cv_train_loader.dataset))
        #print("Val loader size: ", len(cv_val_loader.dataset))
        #print("Test loader size: ", len(cv_test_loader.dataset))
        
        train(model, cv_train_loader, cv_val_loader, criterion, optimizer, num_epochs)
    
    print("\nTesting the model on test dataset")
    test(model, cv_test_loader)
    

class ModelEstimator(BaseEstimator, RegressorMixin): 
    def __init__(self, input_dim, output_dim, nlayer, nhead, hidden_dim, dropout, learning_rate, weight_decay, batch_size): 
        self.input_dim = input_dim 
        self.output_dim = output_dim 
        self.nlayer = nlayer
        self.nhead = nhead
        self.hidden_dim = hidden_dim 
        self.dropout = dropout 
        self.learning_rate = learning_rate 
        self.weight_decay = weight_decay
        self.batch_size = batch_size 
        self.model = None 
        
    def fit(self, X, y): 
        self.model = TransformerModel(self.input_dim, self.output_dim, self.nlayer, self.nhead, self.hidden_dim, self.dropout).to(device)
        criterion = nn.MSELoss() 
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        
        random_state = 42 #Ser random state for reproducibility 
        shuffle_indices = np.random.permutation(len(nw_geno))
        nw_geno_shuffled = nw_geno[shuffle_indices]
        nw_pheno_shuffled = nw_pheno[shuffle_indices] 
        
        X_train, X_test, y_train, y_test = train_test_split(nw_geno_shuffled, nw_pheno_shuffled, test_size=0.2, random_state= random_state)
        
        #print("Train size: ", X_train.shape)
        #print("Test size: ", X_test.shape)
        
        train_subset_size = int(0.8* len(X_train))
        
        X_train_subset = X_train[:train_subset_size]
        y_train_subset = y_train[:train_subset_size]
        
        X_val = X_train[train_subset_size:]
        y_val = y_train[train_subset_size:]
        
        #print("Train subset size: ", X_train_subset.shape)
        #print("Val size: ", X_val.shape)
        
        rsv_train_dataset = GenDataset(X_train_subset, y_train_subset)
        rsv_val_dataset = GenDataset(X_val, y_val)

        
        rsv_train_loader = torch.utils.data.DataLoader(dataset = rsv_train_dataset, batch_size = self.batch_size, shuffle = True) 
        rsv_val_loader = torch.utils.data.DataLoader(dataset = rsv_val_dataset, batch_size = self.batch_size, shuffle = False)
        
        #print("Train loader size: ", len(rsv_train_loader.dataset))
        #print("Val loader size: ", len(rsv_val_loader.dataset))
    
        train(self.model, rsv_train_loader, rsv_val_loader, criterion, optimizer, num_epochs)
        return self 
    
    def score(self, X, y): 
        
        random_state = 42 #Ser random state for reproducibility 
        shuffle_indices = np.random.permutation(len(nw_geno))
        nw_geno_shuffled = nw_geno[shuffle_indices]
        nw_pheno_shuffled = nw_pheno[shuffle_indices] 
        
        X_train, X_test, y_train, y_test = train_test_split(nw_geno_shuffled, nw_pheno_shuffled, test_size=0.2, random_state= random_state)
        
        #Redefine and Create Test Loader
        rsv_test_dataset = GenDataset(X_test, y_test)
        rsv_test_loader = torch.utils.data.DataLoader(dataset = rsv_test_dataset, batch_size = self.batch_size, shuffle = False) 
        #print("Test loader size: ", len(nw_test_loader.dataset))
        
        test(self.model, rsv_test_loader)
        return self
    
    def predict(self, X): 
        self.model.eval()
        with torch.no_grad(): 
            X_tensor = torch.from_numpy(X).to(device)
            outputs = self.model(X_tensor) 
        return outputs.cpu().numpy().squeeze()

"""
if __name__ == "__main__":
    kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
    
    #Define parameter distributions
    params_dist = {
        "batch_size": [50, 100, 150], 
        "hidden_dim": [256, 512, 768, 1024], 
        "dropout": [0.3, 0.4, 0.5], 
        "learning_rate": [1e-5, 1e-4, 1e-3], 
        "weight_decay": [6, 7, 8, 9, 10], 
        "nlayer": [2, 4, 6, 8]}
    
    #Define scorers
    def custom_corrcoef(y_true, y_pred):
        mask = ~np.isnan(y_true.squeeze()) & ~np.isnan(y_pred)
        
        #print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}, mask shape: {mask.shape}")
        
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        #print(f"After masking - y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")    
        
    
        if len(y_true) == 0 or len(y_pred) == 0: 
            print("No valid data points for corrcoef")
            return 0.0
        
        corr_matrix = np.corrcoef(y_true.squeeze(), y_pred)
        
        if np.isnan(corr_matrix).any(): 
            print("NaN values encountered in corr_matrix")
            return 0.0 
        else: 
            corr_coef = corr_matrix[0, 1]
            return corr_coef
            
    #Create scorer objects
    custom_scorer_corr = make_scorer(custom_corrcoef)

    search = RandomizedSearchCV(
        estimator = ModelEstimator(input_dim, output_dim, nlayer, nhead, hidden_dim, dropout, learning_rate, weight_decay, batch_size), 
        param_distributions = params_dist, 
        scoring = {
            "corr_coef" : custom_scorer_corr},
        n_iter = 15, 
        cv = kf,                         
        verbose = 2, 
        n_jobs = 1, 
        random_state = 42, 
        refit = "corr_coef") #refit best model based on correlation coefficient
    
    search.fit(nw_geno, nw_pheno)
    
    print("Best Parameters: ", search.best_params_)
    print("Best Scores: ", search.best_score_)
    #print("RandomizedSearchCV results: \n", search.cv_results_)
    
    end = time.time()
    print("Total time taken", end - start)

    
"""
if __name__ == "__main__": 
    #Train Model
    train_with_cv(model, criterion, optimizer, nw_geno, nw_pheno, num_epochs, batch_size, n_splits = 5)

    #Save Model
    torch.save(model.state_dict(), "nw_model_ckpt.ckpt")

