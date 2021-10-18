import util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # self.layer3 = nn.Linear(32, output_size)

    def forward(self, x):
        x, _ = self.layer1(x)
        s, b, h = x.size()
        x = x.view(s * b, h)  # view函数调整矩阵的形状，类似于reshape
        # x = F.relu(x)
        x = self.layer2(x)
        # x = F.relu(x)
        # x = self.layer3(x)
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


def pred_save(dataloader,model_save_path, model, output_file):
    model.eval()
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    scaler = dataloader['scaler']
    # loss_list = []
    with torch.no_grad():
        X, y = dataloader['test_loader'].get_origin()
        X = scaler.transform(X)

        X = np.expand_dims(X, axis=1)
        y = np.expand_dims(y, axis=1)

        X = torch.tensor(X).to(device)
        y = torch.tensor(y).to(device)

        pred = model(X)
        pred_real = scaler.inverse_transform(pred)

        # print(X.shape)
        # print(y.shape)
        # print(pred_real.shape)
    origin_x = scaler.inverse_transform(X).to('cpu').numpy()
    num_y = y.to('cpu').numpy()
    num_pred = pred_real.to('cpu').numpy()

    i_size = origin_x.shape[2]
    o_size = num_y.shape[2]
    seq_size = origin_x.shape[0] + i_size - 1

    dimen_length = origin_x.shape[1]
    seqs = []
    for dimen in range(dimen_length):
        seq_x = origin_x[0, dimen, :]
        seq_x_mid = origin_x[1:, dimen, -1]
        seq_x = np.append(seq_x, seq_x_mid)
        seq_x = np.append(seq_x, num_y[-1, dimen, :])
        pad = np.zeros([i_size, 1])
        pad_right = np.zeros([o_size - 1, 1])
        seq_y1 = np.append(pad, num_pred[:, dimen, 0])
        seq_y1 = np.append(seq_y1, pad_right)
        # seq_y1 = np.hstack()
        seq_y2 = np.append(pad, num_pred[:, dimen, 1])
        seq_y2 = np.append(seq_y2, pad_right)
        seq_y3 = np.append(pad, num_pred[:, dimen, 2])
        seq_y3 = np.append(seq_y3, pad_right)

        # print(seq_x.shape)
        # print(seq_y1.shape)
        # print(seq_y2.shape)
        # print(seq_y3.shape)

        seq = [seq_x.tolist(), seq_y1.tolist(), seq_y2.tolist(), seq_y3.tolist()]
        seqs.append(seq)

    csv_str = ""
    for seq in seqs:
        seqx = [str(i) for i in range(len(seq[0]))]
        line_str = ",".join(seqx)
        csv_str += line_str
        csv_str += '\n'
        for one in seq:
            # float 转str
            for i in range(len(one)):
                one[i] = "{:.3f}".format(one[i])
            line_str = ",".join(one)
            csv_str += line_str
            csv_str += '\n'
    # seqx = range(len(seqs[0][0]))
    # print(seqx)
    with open(output_file, 'w') as f:
        f.write(csv_str)

    # print(csv_str)


def run_once(root_dir, site_index, factor_index,early_stopping,model_save_path,
             input_length,output_length, epochs, save_pred_csv=False):
    print('################################')
    print('runing site:{},factor:{}'.format(site_index,factor_index))
    site_code = "abcdefghijklmn"
    dataloader = util.load_dataset(root_dir + '/singlesingle/{}{}'.format(factor_index,site_code[site_index]),
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

    if save_pred_csv:
        pred_save(dataloader,model_save_path, model,"data/predict_result/{}{}.csv".format(factor_index,site_code[site_index]))

    loss_result = test(dataloader,model_save_path, model, loss_fn)
    return loss_result


ids_shangban = [ '天宝大水港排涝站','中排渠涝站（天宝）',
              '甘棠溪慧民花园监测点',
        '康山溪金峰花园监测点', '芗城水利局站','中山桥水闸站', '北京路水闸站','九湖监测点','桂林排涝站','上坂']

# 表格中的顺序
factors = ['pH值', '总氮', '总磷', '氨氮', '水温', '浑浊度', '溶解氧', '电导率', '高锰酸盐指数']


if __name__ == '__main__':
    # Get cpu or gpu device for training.
    print('start train...')
    torch.cuda.set_device(1)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    model_save_path = "./data/save_models/simpleLSTM/LSTM.pth"

    input_size = 24
    predict_size = 3
    train_epoch = 150

    ####################### 单因子
    early_stopping = earlystopping.EarlyStopping(patience=30, path=model_save_path, verbose=True)
    res = run_once('data/water/shangban',2, 7, early_stopping,model_save_path, input_size,predict_size,train_epoch,True)
    print("test MAE = {}".format(res))
    ######################## 全站点 全因子实验
    # t1 = time.time()
    # all_factor_result = []
    # for factor in range(9):
    #     site_result = []
    #     for site in range(10):
    #         early_stopping = earlystopping.EarlyStopping(patience=30, path=model_save_path, verbose=True)
    #         res = run_once('data/water/shangban', site, factor,early_stopping,
    #                        model_save_path,input_size,predict_size,train_epoch)
    #         site_result.append(res)
    #     all_factor_result.append(site_result)
    #
    # t2 = time.time()
    #
    # print("--------------------------------------------------")
    # for factor in range(len(all_factor_result)):
    #     line = "{},".format(factors[factor])
    #     for site in range(len(all_factor_result[factor])):
    #         line += "{:.4f},".format(all_factor_result[factor][site])
    #     # print(line)
    #
    #     factor_mean = np.mean(all_factor_result[factor])
    #
    #     print(line + "{:.4f}".format(factor_mean))
    #     # print("---------")
    # print("Total time spent: {:.4f}".format(t2 - t1))
    #########################