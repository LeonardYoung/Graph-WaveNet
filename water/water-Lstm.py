import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

# data_csv = pd.read_csv('./data/mthird2.csv', usecols=[3])[87840:96623]
data_csv = pd.read_csv('data/water/water2020.csv', usecols=[3])[1:8770]
# date_csv = date_csv.flatten()
# 数据预处理
data_csv = data_csv.dropna()  # 滤除缺失数据
dataset = data_csv.values   # 获得csv的值
dataset = dataset.astype('float32')
mean_value = np.mean(dataset)
std_value = np.std(dataset)
# mean_value = 0
# std_value = 1
dataset = list(map(lambda x: (x - mean_value) / std_value, dataset))  # 归一化

predict_input = 24
predict_len = 9


def create_dataset(dataset, num_timesteps_input=24, num_timesteps_output=9):
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(len(dataset) - (
                num_timesteps_input + num_timesteps_output) + 1)]
    dataX, dataY = [], []
    for i, j in indices:
        dataX.append(
            dataset[i: i + num_timesteps_input])
        dataY.append(dataset[i + num_timesteps_input: j])
    return np.array(dataX), np.array(dataY)
torch.cuda.set_device(1)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print("using device:{}".format(device))

# 创建好输入输出
data_X, data_Y = create_dataset(dataset,predict_input,predict_len)


# 划分训练集和测试集，70% 作为训练集
train_size = int(len(data_X) * 0.7)
test_size = len(data_X) - train_size
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]

train_X = train_X.reshape(-1, 1, predict_input)
train_Y = train_Y.reshape(-1, 1, predict_len)
test_X = test_X.reshape(-1, 1, predict_input)
test_Y = test_Y.reshape(-1, 1, predict_len)


test_x = torch.from_numpy(test_X)
test_y = torch.from_numpy(test_Y)


# checkpoint_save_path = './checkpoint/PhArgs.pth'


class lstm(nn.Module):
    def __init__(self, input_size=24,  output_size=9,hidden_size=48, num_layer=2):
        super(lstm, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, output_size)
        # self.layer3 = nn.Linear(128, 64)
        # self.layer4 = nn.Linear(64, 32)
        # self.layer5 = nn.Linear(32, output_size)

    def forward(self, x):
        x, _ = self.layer1(x)
        s, b, h = x.size()
        x = x.view(s * b, h)  # view函数调整矩阵的形状，类似于reshape
        x = self.layer2(x)
        # x = self.layer3(F.relu(x))
        # x = self.layer4(F.relu(x))
        # x = self.layer5(F.relu(x))
        x = x.view(s, b, -1)
        return x

import util
loss_fn = util.masked_mae

# better LSTM
import utils.better_LSTM as better
# model = better.LSTM(24,9).to(device)

model = lstm(predict_input,predict_len)

model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

losses = []
# 开始训练
for e in range(100):

    permutation = np.random.permutation(len(train_X))
    train_x, train_y = train_X[permutation], train_Y[permutation]
    # train_x = xs
    # train_y = ys

    train_x = torch.from_numpy(train_X)
    train_y = torch.from_numpy(train_Y)

    var_x = train_x.to(device)
    var_y = train_y.to(device)

    # 前向传播
    model.train()
    out = model(var_x)
    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    # if (e + 1) % 10 == 0:  # 每 100 次输出结果
    # print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))

    model.eval() # 转换成测试模式

########################
    with torch.no_grad():
        var_data = Variable(test_x).to(device)
        pred_test = model(var_data)
        # 测试集的预测结果
        pred_test_original = pred_test * std_value + mean_value

        # 真实结果
        test_Y_original = test_y * std_value + mean_value
        test_Y_original = Variable(test_Y_original).to(device)

        # mid1 = test_Y_original - pred_test_original
        # mid2 = torch.abs(mid1)
        # # for d in range(predict_len):
        # mid3 = torch.mean(mid2,dim=0)
        #
        # mid2_np = mid2.data.cpu().numpy()
        #
        # print("在第{}个上的mae是{}".format(d+1, mid3))
        # torch.ab
        # mae = torch.mean(torch.abs(test_Y_original - pred_test_original))
        mae = loss_fn(test_Y_original,pred_test_original)

        # x1 = torch.tensor(mae)

        print("Epoch:{},在测试集上的mae是{}".format(e, mae))
############################

    # var_data = Variable(test_x).to(device)
    # pred_test = model(var_data) # 测试集的预测结果
    # pred_test_original = pred_test.view(-1).data.cpu().numpy() * std_value + mean_value
    #
    # test_Y_original = test_y.view(-1).data.cpu().numpy() * std_value + mean_value
    # mae = np.mean(np.absolute(test_Y_original - pred_test_original))
    # print("在测试集上的mae是{}".format(mae))


# watch_start = 11000
# watch_end = 12000
## 画出实际结果和预测的结果

# plt.plot(pred_test_original[watch_start:watch_end], 'r', label='prediction')
# plt.plot(test_Y_original[watch_start:watch_end], 'b', label='real')
# plt.legend(loc='best')
# plt.savefig('fig2.png')
# plt.show()
#
