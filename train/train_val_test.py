import torch
import csv as csv
import torch.nn.functional as F
import torch.distributions
from graph_dataset_gen import Mydataset
from multi_gdn import MGDPR
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

num_nodes, time_steps, num_relation, zeta, diffusion_steps = len(NASDAQ_com_list), 21, 5, 1.001, 7 # zeta = 1.27 sometimes results in nan in objective value, using a smaller one helps training without sacrificing performance.

# Generate datasets
train_dataset = MyDataset(directory, des, market[0], NASDAQ_com_list, sedate[0], sedate[1], time_steps, dataset_type[0])
validation_dataset = MyDataset(directory, des, market[0], NASDAQ_com_list, sedate[0], sedate[1], time_steps, dataset_type[0])
test_dataset = MyDataset(directory, des, market[0], NASDAQ_com_list, sedate[0], sedate[1], time_steps, dataset_type[0])

# Define model
diffusion_dims = [num_relation * time_steps, 128, 256, 512, 512, 512, 256, 128, 64]
ret_in_dim = [128, 256, 512, 512, 512, 256, 128, 64]
ret_inter_dim = [512, 512, 512, 512, 512, 512, 512, 512]
ret_hidden_dim = [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]
ret_out_dim = [256, 256, 256, 256, 256, 256, 256, 256]
post_pro = [256, 64, 2]

# Define model
model = MGDPR(num_nodes, diffusion_dims, ret_in_dim, ret_inter_dim, ret_hidden_dim, ret_out_dim,
              post_pro, num_relation, diffusion_steps, zeta)

# Pass model and datasets to GPU
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

# Define training process & validation process & testing process
epochs = 10000

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
        correct += int((out == C).sum())
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
