import util
import numpy as np
import torch
import torch.nn as nn


dataloader = util.load_dataset('data/water/singlesingle/0a',64,64,64,False)
# Get cpu or gpu device for training.
torch.cuda.set_device(1)
device = "cuda:1" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(24, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 3)
#         )
#
#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits
#
# model = NeuralNetwork().to(device)

class lstm(nn.Module):
    def __init__(self, input_size=24, hidden_size=8, output_size=3, num_layer=1):
        super(lstm, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.layer1(x)
        s, b, h = x.size()
        x = x.view(s * b, h)  # view函数调整矩阵的形状，类似于reshape
        x = self.layer2(x)
        x = x.view(s, b, -1)
        return x

# better LSTM
# import utils.better_LSTM as better
# model = better.LSTM(24,3).to(device)


model = lstm().to(device)
model = model.double()
print(model)

loss_fn = util.masked_mae
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.RMSprop(model.parameters(),lr=0.001)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(dataloader, model, loss_fn, optimizer):
    # size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader['train_loader'].get_iterator()):
        X = np.expand_dims(X,axis=1)
        y = np.expand_dims(y,axis=1)
        # print(X.shape)
        X = torch.tensor(X).to(device)
        y = torch.tensor(y).to(device)

        # Compute prediction error
        pred = model(X)
        # print(pred.shape)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(dataloader, model, loss_fn):
    model.eval()
    loss_list = []
    with torch.no_grad():
        for X, y in dataloader['test_loader'].get_iterator():
            X = np.expand_dims(X, axis=1)
            y = np.expand_dims(y, axis=1)

            X = torch.tensor(X).to(device)
            y = torch.tensor(y).to(device)


            pred = model(X)
            loss_list.append(loss_fn(pred, y).item())

    loss_result = np.mean(loss_list)
    return loss_result

import time
epochs = 1500
t1 = time.time()
for t in range(epochs):
    # print(f"Epoch {t+1}\n-------------------------------")
    train(dataloader, model, loss_fn, optimizer)
    loss_result = test(dataloader, model, loss_fn)
    print("Epoch:{},loss:{}".format(t+1,loss_result))

t2 = time.time()
print("Total time spent: {:.4f}".format(t2-t1))

