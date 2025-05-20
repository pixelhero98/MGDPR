import torch
import csv as csv
import torch.nn.functional as F
import torch.distributions
from graph_dataset_gen import MyDataset, MultiIndexDataset
from Multi_GDNN import MGDPR
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

# Path to the combined multi-index CSV file (ticker, date as indices)
multi_csv_path = '/path/to/multiindex.csv'

# Generate datasets using the MultiIndexDataset to avoid look-ahead bias
train_dataset = MultiIndexDataset(multi_csv_path, des, NASDAQ_com_list,
                                  sedate[0], sedate[1], 19, dataset_type[0])
validation_dataset = MultiIndexDataset(multi_csv_path, des, NASDAQ_com_list,
                                       val_sedate[0], val_sedate[1], 19, dataset_type[1])
test_dataset = MultiIndexDataset(multi_csv_path, des, NASDAQ_com_list,
                                 test_sedate[0], test_sedate[1], 19, dataset_type[2])

# Define model (these can be tuned)
n = len(NASDAQ_com_list) # number of companies in NASDAQ

d_layers, num_nodes, time_steps, num_relation, gamma, diffusion_steps = 6, n, 21, 5, 2.5e-4, 7

diffusion_layers = [time_steps, 3 * time_steps, 4 * time_steps, 5 * time_steps, 5 * time_steps, 6 * time_steps, 5 * time_steps]

retention_layers = [num_relation*3*n, num_relation*5*n, num_relation*4*n,
                    num_relation*4*n, num_relation*5*n, num_relation*5*n,
                    num_relation*5*n, num_relation*5*n, num_relation*5*n,
                    num_relation*5*n, num_relation*5*n, num_relation*5*n,
                    num_relation*6*n, num_relation*5*n, num_relation*5*n,
                    num_relation*5*n, num_relation*5*n, num_relation*5*n]


ret_linear_layers_1 = [time_steps * num_relation, time_steps * num_relation,
                     time_steps * num_relation * 5, time_steps * num_relation,
                     time_steps * num_relation * 6, time_steps * num_relation,
                     time_steps * num_relation * 6, time_steps * num_relation,
                     time_steps * num_relation * 6, time_steps * num_relation,
                     time_steps * num_relation * 6, time_steps * num_relation]


ret_linear_layers_2 = [time_steps * num_relation * 5, time_steps * num_relation * 5,
                     time_steps * num_relation * 6, time_steps * num_relation * 6,
                     time_steps * num_relation * 6, time_steps * num_relation * 6,
                     time_steps * num_relation * 6, time_steps * num_relation * 6,
                     time_steps * num_relation * 6, time_steps * num_relation * 6,
                     time_steps * num_relation * 6, time_steps * num_relation * 6]

mlp_layers = [num_relation * 5 * time_steps + time_steps * num_relation, 128, 2]

# Define model
model = MGDPR(diffusion_layers, retention_layers, ret_linear_layers_1, ret_linear_layers_2, mlp_layers, d_layers,
              num_nodes, time_steps, num_relation, gamma, diffusion_steps)

# Pass model and datasets to GPU
model = model.to(device)

# Define optimizer and objective function


def theta_regularizer(theta):
    row_sums = torch.sum(theta.to(device), dim=-1)
    ones = torch.ones_like(row_sums)
    return torch.sum(torch.abs(row_sums - ones))


#def D_gamma_regularizer(D_gamma):
    #upper_tri = torch.triu(D_gamma, diagonal=1)
    #return torch.sum(torch.abs(upper_tri))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Define training process & validation process & testing process
epochs = 10000
model.reset_parameters()

# Training and validating
for epoch in range(epochs):
    model.train()
    objective_total = 0
    correct = 0
    total = 0

    for sample in train_dataset: # Recommend to update every sample, full batch training can be time-consuming
        X = sample['X'].to(device)  # node feature tensor
        A = sample['A'].to(device)  # adjacency tensor
        C = sample['Y'].long()
        C = C.to(device)  # label vector

        optimizer.zero_grad()
        out = model(X, A)
        objective = F.cross_entropy(out, C) #+ theta_regularizer(model.theta) regularization may resultvery slow learning process, optional usage.
        objective.backward()
        optimizer.step()
        objective_total += objective.item()

    # If performance progress of the model is required
        out = out.argmax(dim=1)
        correct += int((out == C).sum()).item()
        total += C.shape[0]
        if epoch % 1 == 0:
          print(f"Epoch {epoch}: loss={objective_total:.4f}, acc={correct / total:.4f}")

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
print(mcc / len(validation_dataset))

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
    torch.save(model, dir_path() + 'your_dataset_name/model')
