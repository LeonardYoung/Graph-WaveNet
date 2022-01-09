import util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from utils import earlystopping
import os
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
place = 'shangban'
fac_index = 8

class lstm(nn.Module):
    def __init__(self, input_size=24, output_size=9, hidden_size=64,  num_layer=2):
        super(lstm, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, output_size)
        # self.layer3 = nn.Linear(32, output_size)

    def forward(self, x):
        # x = nn.BatchNorm2d(x)
        x, _ = self.layer1(x)
        s, b, h = x.size()
        x = x.view(s * b, h)  # view函数调整矩阵的形状，类似于reshape
        # x = F.relu(x)
        x = self.layer2(x)
        # x = F.relu(x)
        # x = self.layer3(x)
        x = x.view(s, b, -1)
        return x


class CNN_LSTM(nn.Module):
    def __init__(self, input_size=24, output_size=3, hidden_size=64,  num_layer=2):
        super(CNN_LSTM, self).__init__()

        self.start_cnn = nn.Conv1d(in_channels=1,out_channels=16,kernel_size=1)

        self.cnn1 = nn.Conv1d(in_channels=16, out_channels=32,kernel_size=1)
        self.cnn2 = nn.Conv1d(in_channels=32,out_channels=32,kernel_size=1)

        self.lstm1 = nn.LSTM(input_size=24, hidden_size=24, num_layers=2)
        # self.lstm1 = nn.GRU(input_size=24, hidden_size=64, num_layers=2)
        self.dnn = nn.Linear(24, 3)

    def forward(self, x):
        x = self.start_cnn(x)
        # print(x.shape)
        x = self.cnn1(x)
        # x = torch.relu(x)
        # x = F.dropout(x, 0.3, training=self.training)
        x = self.cnn2(x)
        # x = F.dropout(x, 0.3, training=self.training)
        # x = torch.relu(x)


        x = self.lstm1(x)[0]
        x = self.dnn(x)

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

from sklearn import metrics as sk_metrics
def test(dataloader,model_save_path, model, loss_fn):
    model.eval()
    model.load_state_dict(torch.load(model_save_path))
    scaler = dataloader['scaler']
    # mae_list = []
    # mape_list = []
    with torch.no_grad():
        x_test = dataloader['x_test']
        y_test = dataloader['y_test']
        # for X, y in dataloader['test_loader'].get_iterator():
        x_test = scaler.transform(x_test)

        x_test = np.expand_dims(x_test, axis=1)
        y_test = np.expand_dims(y_test, axis=1)

        x_test = torch.tensor(x_test).to(device)
        y_test = torch.tensor(y_test).to(device)

        pred = model(x_test)
        pred = scaler.inverse_transform(pred)

        # 保存预测值到文件
        save_root = f"data/output/{place}/y/LSTM/{fac_index}"
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        pred_np = pred.to('cpu').numpy()
        realy_np = y_test.to('cpu').numpy()
        np.savez_compressed(
            os.path.join(save_root, f"out.npz"),
            y_pred=pred_np,
            y_test=realy_np
        )

        metrics = util.metric(pred, y_test)

        for step in range(3):
            y_test_t = y_test[..., step].to('cpu').numpy()
            y_pred_t = pred[..., step].to('cpu').numpy()

            r2 = sk_metrics.r2_score(y_test_t, y_pred_t)
            mae = sk_metrics.mean_absolute_error(y_test_t, y_pred_t)
            rmse = sk_metrics.mean_squared_error(y_test_t, y_pred_t) ** 0.5
            mape = sk_metrics.mean_absolute_percentage_error(y_test_t, y_pred_t)

            print(f'MAE:{mae:.3f},RMSE:{rmse:.3f},MAPE:{mape:.3f},R2:{r2:.3f}')

        # for step in range(3):
        #     metric = util.metric(pred[...,step],y_test[...,step])
        # # MAE
        # mae_list.append(metrics[0])
        # # MAPE
        # mape_list.append(metrics[1])

    return metrics[0],metrics[1]


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

# import water.SVR.data_generate as data_generate
def run_once(root_dir, factor_index,early_stopping,model_save_path,
             input_length,output_length, epochs, save_pred_csv=False):
    print('################################')
    print('runing factor:{}'.format(factor_index))
    place = 'shangban'

    data = util.load_dataset(f'data/water/{place}/singleFac/{factor_index}', 64,64,64)
    for category in ['train', 'val', 'test']:
        # 去掉时间维
        data['x_' + category] = data['x_' + category][..., 0]
        data['y_' + category] = data['y_' + category][..., 0]

        # 将不同站点的数据拼接在一起。
        site_num = data['x_' + category].shape[2]
        x_concate = []
        y_concate = []
        for i in range(site_num):
            x_concate.append(data['x_' + category][:, :, i])
            y_concate.append(data['y_' + category][..., i])
        x_concate = np.concatenate(x_concate)
        y_concate = np.concatenate(y_concate)
        data['x_' + category] = x_concate
        data['y_' + category] = y_concate
        data[category + '_loader'].set_xy(data['x_' + category],data['y_' + category])


    # model = CNN_LSTM(input_length, output_length).to(device)
    model = lstm(input_length, output_length).to(device)
    model = model.double()

    loss_fn = util.masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for t in range(epochs):
        # print(f"Epoch {t+1}\n-------------------------------")
        train(data, model, nn.MSELoss(), optimizer)
        loss_result = validate(data, model, loss_fn)
        if t % 10 == 9:
            print("Epoch:{},validate loss:{}".format(t + 1, loss_result))
        early_stopping(loss_result, model)
        if early_stopping.early_stop:
            print("Early stopping.")
            break

    if save_pred_csv:
        pred_save(data,model_save_path, model,"data/predict_result/{}{}.csv".format(factor_index,site_code[site_index]))

    loss_result = test(data,model_save_path, model, loss_fn)
    return loss_result


ids_shangban = [ '天宝大水港排涝站','中排渠涝站（天宝）',
              '甘棠溪慧民花园监测点',
        '康山溪金峰花园监测点', '芗城水利局站','中山桥水闸站', '北京路水闸站','九湖监测点','桂林排涝站','上坂']

# 表格中的顺序
factors = ['pH值', '总氮', '总磷', '氨氮', '水温', '浑浊度', '溶解氧', '电导率', '高锰酸盐指数']


import random
if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Get cpu or gpu device for training.
    print('start train...')

    # torch.cuda.set_device(1)
    # device = "cuda:1" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda:1"
        torch.cuda.set_device(1)
    else:
        device = "cpu"

    print("Using {} device".format(device))
    model_save_path = f"data/output/{place}/model/LSTM/model.pth"

    input_size = 24
    predict_size = 3
    train_epoch = 1000

    # ###################### 单因子
    early_stopping = earlystopping.EarlyStopping(patience=50, path=model_save_path, verbose=True)
    res = run_once('data/water/shangban/singlesingle',fac_index,  early_stopping,model_save_path, input_size,predict_size,train_epoch)
    print("test MAE = {},MAPE={}".format(res[0],res[1]))

    # ################单因子全站点
    # factor_index = 0
    # t1 = time.time()
    # site_result = []
    # for site in range(10):
    #     early_stopping = earlystopping.EarlyStopping(patience=50, path=model_save_path, verbose=True)
    #     res = run_once('data/water/shangban/singlesingle', site, factor_index,early_stopping,
    #                    model_save_path,input_size,predict_size,train_epoch,True)
    #     site_result.append(res[0])
    #
    # t2 = time.time()
    #
    # # 打印结果
    # print("-------------------MAE-------------------------------")
    # line = ""
    # for site in range(len(site_result)):
    #     # print(all_factor_result[factor][site])
    #     line += "{:.4f},".format(site_result[site])
    # print(line)
    #
    # factor_mean = np.mean(site_result[:])
    # print("{:.4f}".format(factor_mean))


    # ####################### 全站点 全因子实验
    # t1 = time.time()
    # all_factor_result = []
    # for factor in range(9):
    #     site_result = []
    #     for site in range(10):
    #         early_stopping = earlystopping.EarlyStopping(patience=30, path=model_save_path, verbose=True)
    #         res = run_once('data/water/shangban/singlesingle', site, factor,early_stopping,
    #                        model_save_path,input_size,predict_size,train_epoch,True)
    #         site_result.append(res)
    #     all_factor_result.append(site_result)
    #
    # t2 = time.time()
    #
    # # 打印结果
    # print("-------------------MAE-------------------------------")
    # for factor in range(len(all_factor_result)):
    #     line = "{},".format(factors[factor])
    #     for site in range(len(all_factor_result[factor])):
    #         # print(all_factor_result[factor][site])
    #         line += "{:.4f},".format(all_factor_result[factor][site][0])
    #     # factor_mean = np.mean(all_factor_result[factor][:][0])
    #     # print(line + "{:.4f}".format(factor_mean))
    #     # print(all_factor_result[factor][:][0])
    #     print(line)
    # # print("Total time spent: {:.4f}".format(t2 - t1))
    # print("--------------------MAPE------------------------------")
    # for factor in range(len(all_factor_result)):
    #     line = "{},".format(factors[factor])
    #     for site in range(len(all_factor_result[factor])):
    #         line += "{:.4f},".format(all_factor_result[factor][site][1])
    #     print(line)
    #########################