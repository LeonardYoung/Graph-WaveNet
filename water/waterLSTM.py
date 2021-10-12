import util
import numpy as np
import torch
import torch.nn as nn
import time
from utils import earlystopping

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
    def __init__(self, input_size=24, output_size=9, hidden_size=64,  num_layer=2):
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


def train(dataloader, model, loss_fn, optimizer):
    #
    scaler = dataloader['scaler']
    dataloader['train_loader'].shuffle()
    model.train()
    for batch, (X, y) in enumerate(dataloader['train_loader'].get_iterator()):
        X = scaler.transform(X)

        X = np.expand_dims(X,axis=1)
        y = np.expand_dims(y,axis=1)
        # print(X.shape)
        X = torch.tensor(X).to(device)
        y = torch.tensor(y).to(device)

        # Compute prediction error
        pred = model(X)

        pred_real = scaler.inverse_transform(pred)
        # print(pred.shape)
        loss = loss_fn(pred_real, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(dataloader, model, loss_fn):
    model.eval()
    scaler = dataloader['scaler']
    loss_list = []
    with torch.no_grad():
        for X, y in dataloader['val_loader'].get_iterator():
            X = scaler.transform(X)

            X = np.expand_dims(X, axis=1)
            y = np.expand_dims(y, axis=1)

            X = torch.tensor(X).to(device)
            y = torch.tensor(y).to(device)

            pred = model(X)
            pred_real = scaler.inverse_transform(pred)

            loss_list.append(loss_fn(pred_real, y).item())

    loss_result = np.mean(loss_list)
    return loss_result


def test(dataloader,model_save_path, model, loss_fn):
    model.eval()
    model.load_state_dict(torch.load(model_save_path))
    scaler = dataloader['scaler']
    loss_list = []
    with torch.no_grad():
        for X, y in dataloader['test_loader'].get_iterator():
            X = scaler.transform(X)

            X = np.expand_dims(X, axis=1)
            y = np.expand_dims(y, axis=1)

            X = torch.tensor(X).to(device)
            y = torch.tensor(y).to(device)

            pred = model(X)
            pred_real = scaler.inverse_transform(pred)

            loss_list.append(loss_fn(pred_real, y).item())

    loss_result = np.mean(loss_list)
    return loss_result


def run_once(site_index, factor_index,early_stopping,model_save_path, input_length,output_length, epochs= 150):
    print('################################')
    print('runing site:{},factor:{}'.format(site_index,factor_index))
    site_code = "abcdefghijklmn"
    dataloader = util.load_dataset('data/water/singlesingle/{}{}'.format(factor_index,site_code[site_index]),
                                   64, 64, 64, False)

    model = lstm(input_length, output_length).to(device)
    model = model.double()

    loss_fn = util.masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for t in range(epochs):
        # print(f"Epoch {t+1}\n-------------------------------")
        train(dataloader, model, nn.MSELoss(), optimizer)
        loss_result = validate(dataloader, model, loss_fn)
        if t % 10 == 9:
            print("Epoch:{},validate loss:{}".format(t + 1, loss_result))
        early_stopping(loss_result, model)
        if early_stopping.early_stop:
            print("Early stopping.")
            break

    loss_result = test(dataloader,model_save_path, model, loss_fn)
    return loss_result


if __name__ == '__main__':
    # Get cpu or gpu device for training.
    torch.cuda.set_device(1)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    model_save_path = "./data/save_models/simpleLSTM/LSTM.pth"

    ####################### 单因子
    # early_stopping = earlystopping.EarlyStopping(patience=30, path=model_save_path, verbose=True)
    # res = run_once(10, 0, early_stopping,model_save_path, 24, 9, 150)
    # print("test MAE = {}".format(res))

    ######################## 全站点 全因子实验
    t1 = time.time()
    all_factor_result = []
    for factor in range(9):
        site_result = []
        for site in range(11):
            early_stopping = earlystopping.EarlyStopping(patience=40, path=model_save_path, verbose=True)
            res = run_once(site, factor,early_stopping,model_save_path,24,9,150)
            site_result.append(res)
        all_factor_result.append(site_result)

    t2 = time.time()

    print("--------------------------------------------------")
    for factor in range(len(all_factor_result)):
        line = "factor:{}  ".format(factor)
        for site in range(len(all_factor_result[factor])):
            line += "{:.4f},".format(all_factor_result[factor][site])
        print(line)

        factor_mean = np.mean(all_factor_result[factor])

        print("factor_mean={:.4f}".format(factor_mean))
        print("---------")
    print("Total time spent: {:.4f}".format(t2 - t1))