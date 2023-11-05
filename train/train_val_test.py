import torch
import numpy as np
import torch.nn.functional as F
import torch.distributions
import matplotlib.pyplot as plt
import random
import seaborn as sns
import sklearn.preprocessing as skp
from graph_dataset_gen import Mydataset
from Multi_GDNN import MGDPR
from torch_geometric.logging import log
from matplotlib import cm
from matplotlib import axes
from sklearn.metrics import matthews_corrcoef, f1_score

# Configure the device for running the model on GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configure the default variables // # these can be tuned // # examples
sedate = ['2013-01-01', '2014-12-31']  # these can be tuned
val_sedate = ['2015-01-01', '2015-06-30'] # these can be tuned
test_sedate = ['2015-07-01', '2017-12-31'] # these can be tuned
market = ['NASDAQ', 'NYSE', 'SSE'] # can be changed
dataset_type = ['Train', 'Validation', 'Test']
com_path = ['/content/drive/MyDrive/Raw_Data/Stock_Markets/NYSE_NASDAQ/NASDAQ.csv',
            '/content/drive/MyDrive/Raw_Data/Stock_Markets/NYSE_NASDAQ/NYSE.csv',
            '/content/drive/MyDrive/Raw_Data/Stock_Markets/NYSE_NASDAQ/NYSE_missing.csv']
des = '/content/drive/MyDrive/Raw_Data/Stock_Markets/NYSE_NASDAQ/raw_stock_data/stocks_indicators/data'
directory = "/content/drive/MyDrive/Raw_Data/Stock_Markets/NYSE_NASDAQ/raw_stock_data/stocks_indicators/data/google_finance"

NASDAQ_com_list = []
NYSE_com_list = []
NYSE_missing_list = []
com_list = [NASDAQ_com_list, NYSE_com_list, NYSE_missing_list]
for idx, path in enumerate(com_path):
    with open(path) as f:
        file = csv.reader(f)
        for line in file:
            com_list[idx].append(line[0])  # append first element of line if each line is a list
NYSE_com_list = [com for com in NYSE_com_list if com not in NYSE_missing_list]

# Define model (these can be tuned)
d_layers, num_nodes, time_steps, num_relation, gamma, diffusion_steps = 6, 1026, 21, 5, 2.5e-4, 7

diffusion_layers = [time_steps, 63, 84, 105, 105, 126, 105]

retention_layers = [3078, 1024, 4104, 4104, 2048, 5130,
                    5130, 2048, 6156, 5130, 2048, 6156,
                    6156, 2048, 5130, 5130, 4096, 9234]

ret_linear_layers = [21, 84, 105, 126, 126, 105, 189]

mlp_layers = [189, 256, 2]

# Generate datasets
train_dataset = MyDataset(directory, des, market[0], NASDAQ_com_list, sedate[0], sedate[1], 21, dataset_type[0])
validation_dataset = MyDataset(directory, des, market[0], NASDAQ_com_list, sedate[0], sedate[1], 21, dataset_type[0])
test_dataset = MyDataset(directory, des, market[0], NASDAQ_com_list, sedate[0], sedate[1], 21, dataset_type[0])

# Define model
model = MGDPR(diffusion_layers, retention_layers, ret_linear_layers, mlp_layers, d_layers,
              num_nodes, time_steps, num_relation, gamma, diffusion_steps)

# Pass model and datasets to GPU
model = model.to(device)

# Define optimizer and objective function
def theta_regularizer(theta):
    row_sums = torch.sum(theta, dim=-1)
    ones = torch.ones_like(row_sums)
    return torch.sum(torch.abs(row_sums - ones))

#def D_gamma_regularizer(D_gamma):
 #   upper_tri = torch.triu(D_gamma, diagonal=1)
  #  return torch.sum(torch.abs(upper_tri))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Define evaluation metrics
# ACC, MCC, and F1

# Define training process & validation process & testing process
epochs = 1000
model.reset_parameters()

# Training
for epoch in range(epochs):
    model.train()

    objective_total = 0
    acc = 0

    for sample in train_dataset:
        X = sample['X'].to(device)  # node feature tensor
        A = sample['A'].to(device)  # adjacency tensor
        C = sample['Y'].long()
        C = C.to(device)  # label vector

        objective = F.cross_entropy(model(X, A), C)
        objective += theta_regularizer(model.theta)

        objective_total += objective

    objective_average = objective_total / len(train_dataset)
    objective_average.backward()
    optimizer.step()
    optimizer.zero_grad()

    # If performance progress of the model is required
    model.eval()
    for sample in train_dataset:
        X = sample['X'].to(device)  # node feature tensor
        A = sample['A'].to(device)  # adjacency tensor
        C = sample['Y'].long()
        C = C.to(device)  # label vector

        out = model(X, A).argmax(dim=1)
        acc += int((out == C).sum())


    if epoch % 10 == 0:
        print(f'Epoch {epoch}: {objective_average.item()}')
        print('ACC: ', acc / ( len(train_dataset) * C.shape[0]))

# Validation
model.eval()

acc = 0
f1 = 0
mcc = 0

for idx, sample in enumerate(validation_dataset):

    X = sample['X']  # node feature tensor
    A = sample['A']  # adjacency tensor
    C = sample['Y']  # label vector
    out = model(X, A).argmax(dim=1)

    acc += int((out == C).sum())
    f1 += f1_score(C, out.cpu().numpy())
    mcc += matthews_corrcoef(C, out.cpu().numpy())

print(acc / (len(validation_dataset) * C.shape[0]))
print(f1 / len(validation_dataset))
print(mcc/ len(validation_dataset))

# Test
acc = 0
f1 = 0
mcc = 0

for idx, sample in enumerate(test_dataset):
    X = sample['X']  # node feature tensor
    A = sample['A']  # adjacency tensor
    C = sample['Y']  # label vector
    out = model(X, A).argmax(dim=1)

    acc += int((out == C).sum())
    f1 += f1_score(C, out.cpu().numpy())
    mcc += matthews_corrcoef(C, out.cpu().numpy())

print(acc / (len(test_dataset) * C.shape[0]))
print(f1 / len(test_dataset))
print(mcc / len(test_dataset))

# save model to the directory
if int(input('save model? (1/0)?')) == 1:
    torch.save(model, dir_path() + 'NASDAQ/model' + '_' + str(ii) + '_' + str(jj) + '_' + str(kk))
