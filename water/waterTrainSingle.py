# 在终端运行需要
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer
from utils import earlystopping
from torchmetrics import MeanAbsoluteError
from sklearn import metrics as sk_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:1',help='')
# parser.add_argument('--adjlearn',type=str,default='GLM',help='adj learn algorithm')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=11,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='./data/output/model/',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
import water.config as Config

args = parser.parse_args()


def train(engine,dataloader):
    dataloader['train_loader'].shuffle()
    loss_epoch = []
    for step, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
        train_x = torch.Tensor(x).to(engine.device).transpose(1, 3)
        train_y = torch.Tensor(y).to(engine.device).transpose(1, 3)
        loss_batch,_,_ = engine.train(train_x, train_y[:, 0:-1, :, :])
        loss_epoch.append(loss_batch)
    return np.mean(loss_epoch)

def validate(engine,dataloader):
    for step, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
        test_x = torch.Tensor(x).to(engine.device).transpose(1, 3)
        test_y = torch.Tensor(y).to(engine.device).transpose(1, 3)
        metrics = engine.eval(test_x, test_y[:, 0:-1, :, :])
        return metrics


def test(engine,dataloader,model_path):

    engine.model.load_state_dict(torch.load(model_path))
    scaler = dataloader['scaler']

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(engine.device)
    if args.in_dim == 2:
        realy = realy.transpose(1, 3)[:, 0, :, :]
    else:
        realy = realy.transpose(1, 3)[:, 0:-1, :, :]

    engine.model.eval()
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(engine.device).transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    print("Training finished")
    # print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    # 打印出邻接矩阵
    if args.gcn_bool and engine.model.adj is not None:
        adj = engine.model.adj.to('cpu').numpy()

        adj_min = np.min(adj)
        adj_max = np.max(adj)
        adj_avg = np.mean(adj)
        print(adj)
        print('邻接矩阵：min={:.3f},max={:.3f},avg={:.3f}'.format(adj_min,adj_max,adj_avg))

        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                print(adj[i][j], end=',')
            print('')

        # 保存下来
        if Config.fac_single:
            save_root = f"data/output/{Config.place}/adj/singleWaveNet/{Config.out_dir}/{Config.fac_index}"
        else:
            save_root = f"data/output/{Config.place}/adj/multiWaveNet/{Config.out_dir}/{Config.fac_index}"
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        np.savez_compressed(
            f'{save_root}/adj.npz',
            adj=adj,
        )

    # 单维
    if args.in_dim == 2:
        # 计算r，
        # yhat,realy

        # 保存预测值
        preds = scaler.inverse_transform(yhat)
        if Config.fac_single:
            save_root = f"data/output/{Config.place}/y/singleWaveNet/{Config.out_dir}/{Config.fac_index}"
        else:
            save_root = f"data/output/{Config.place}/y/multiWaveNet/{Config.out_dir}/{Config.fac_index}"
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        pred_np = preds.to('cpu').numpy()
        realy_np = realy.to('cpu').numpy()
        np.savez_compressed(
            os.path.join(save_root, f"out.npz"),
            y_pred=pred_np,
            y_test=realy_np
        )


        # 按照步长计算误差
        # MAE_object = MeanAbsoluteError()
        # MAE_object = MAE_object.to(Config.device)
        # mae_err = MAE_object(preds, realy)
        # print(f"MAE_ERR={mae_err}")
        #
        # for i in range(3):
        #     MAE_object = MeanAbsoluteError()
        #     MAE_object = MAE_object.to(Config.device)
        #     mae_err = MAE_object(preds[:,:,i], realy[:,:,i])
        #     print(f"step={i+1},MAE_ERR={mae_err}")
        #
        text_output = []

        amae = []
        amape = []
        armse = []
        ar2 = []
        for i in range(3):
            pred = scaler.inverse_transform(yhat[:, :, i])
            real = realy[:, :, i]
            metrics = util.metric(pred, real)
            r2 = sk_metrics.r2_score(real.to('cpu').numpy(),pred.to('cpu').numpy())
            log = f'horizon {i + 1}, Test MAE: {metrics[0]:.4f}, Test MAPE: {metrics[1]:.4f}, Test RMSE: {metrics[2]:.4f}, Test R2: {r2:.4f}'
            print(log)
            text_output.append(log)
            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])
            ar2.append(r2)

        log = f'On average over 3 horizons, Test MAE: {np.mean(amae):.4f}, Test MAPE: {np.mean(amape):.4f},' \
              f' Test RMSE: {np.mean(armse):.4f}, Test R2: {np.mean(ar2):.4f}'
        print(log)
        text_output.append(log)
        # torch.save(engine.model.state_dict(),f'{args.save}_exp{}.pth' )

        # 保存结果到文件
        if Config.fac_single:
            save_root = f"data/output/{Config.place}/text/singleWaveNet/{Config.out_dir}/{Config.fac_index}"
        else:
            save_root = f"data/output/{Config.place}/text/multiWaveNet/{Config.out_dir}/{Config.fac_index}"
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        fh = open(f'{save_root}/result.txt', 'w', encoding='utf-8')
        fh.write('\n'.join(text_output))
        fh.close()

        # # 按照站点分别计算
        # amae = []
        # amape = []
        # armse = []
        # for i in range(args.num_nodes):
        #     pred = scaler.inverse_transform(yhat[:, i, :])
        #     real = realy[:, i, :]
        #     metrics = util.metric(pred, real)
        #     log = 'Evaluate model on site {:d} , Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        #     print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        #     text_output.append(log)
        #     amae.append(metrics[0])
        #     amape.append(metrics[1])
        #     armse.append(metrics[2])
        #
        # log = 'On average over all site, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        # print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
        # text_output.append(log)
        return np.mean(amae), np.mean(amape), np.mean(armse)
    # 多维
    else:
        pred = scaler.inverse_transform(yhat)
        real = realy
        metrics = util.metric(pred, real)
        log = 'Evaluate model on  Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(metrics[0], metrics[1], metrics[2]))


def run_once():

    # load data
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device).to(torch.float32) for i in adj_mx]
    # supports = [adj_mx[-1]]

    # print(args)
    print(f'gcn_bool={args.gcn_bool}')
    # print(f'adj_learn={args.adjlearn}')
    print(f'data={args.data}')
    Config.print_all()

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    # 只保留初始矩阵
    supports = [supports[-1]]
    if args.aptonly:
        supports = None

    if Config.fac_single:
        save_root = f"data/output/{Config.place}/model/singleWaveNet/{Config.out_dir}/{Config.fac_index}"
    else:
        save_root = f"data/output/{Config.place}/model/multiWaveNet/{Config.out_dir}/{Config.fac_index}"
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    model_save_path = f'{save_root}/model.pth'
    early_stopping = earlystopping.EarlyStopping(patience=Config.patience, path=model_save_path, verbose=True)

    engine = trainer( scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit)

    print("start training...",flush=True)

    train_loss = []
    val_loss = []
    for e in range(1,args.epochs+1):
        train_loss_epoch = train(engine, dataloader)
        train_loss.append(train_loss_epoch)

        val_loss_epoch,_,_ =validate(engine, dataloader)
        val_loss.append(val_loss_epoch)
        print("Epoch:{},validate loss:{}".format(e, val_loss_epoch))

        early_stopping(val_loss_epoch,engine.model)
        if early_stopping.early_stop:
            print("Early stopping.")
            break

    # 保存loss
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    if Config.fac_single:
        save_root = f"data/output/{Config.place}/loss/singleWaveNet/{Config.out_dir}/{Config.fac_index}"
    else:
        save_root = f"data/output/{Config.place}/loss/multiWaveNet/{Config.out_dir}/{Config.fac_index}"
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    np.savez_compressed(
        os.path.join(save_root, f"loss.npz"),
        train_loss=train_loss,
        val_loss=val_loss
    )

    return test(engine,dataloader,model_save_path)


import random
if __name__ == "__main__":
    place = Config.place

    # set seed
    random.seed(Config.seed)
    np.random.seed(Config.seed)
    torch.manual_seed(Config.seed)
    torch.cuda.manual_seed(Config.seed)

    args.aptonly = True
    args.addaptadj = True
    args.randomadj = True
    args.adjtype = 'doubletransition'
    # 输出维度
    args.seq_length = 3
    # 输入维度（包括时间维度）
    args.in_dim = 2
    args.device = Config.device
    args.num_nodes = Config.num_nodes # 图节点数

    # args.device = 'cpu'
    args.gcn_bool = Config.gcn_bool

    if Config.adj_learn_type == 'weigthedOnly':
        args.aptonly = False
        args.addaptadj = False

    args.epochs = Config.epoch
    if Config.fac_single:
        # ######  单因子全站点实验参数
        args.data = f'data/water/{place}/singleFac/{Config.fac_index}'
        args.adjdata = f'data/water/{place}/adjs/adj_all_one.pkl'

    else:
        # ######  多因子实验参数（每个因子是一个站点）
        args.data = f'data/water/{place}/multiFac'
        args.adjdata = f'data/water/{place}/adjs/adj_all_one.pkl'
        # 图节点数
        args.num_nodes = 60 if Config.place == 'shangban' else 42

    ######  全因子多站点实验参数
    # args.epochs = 1000
    # args.data = f'data/water/{place}/all'
    # args.adjdata = f'data/water/{place}/adjs/adj_all_one.pkl'
    # # 输入维度（包括时间维度）
    # args.in_dim = 7


    # ############跑1次


    t1 = time.time()
    run_once()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))


    # ############跑5次
    # t1 = time.time()
    # mae_list = []
    # for i in range(5):
    #     print("data::{},running {} st".format(args.data,i+1))
    #     mae = run_once()
    #     print(mae)
    #     mae_list.append(mae)
    # print(mae_list)
    # mae_list = np.array(mae_list)
    # print("TOTAL MEAN MAE={}".format(np.mean(mae_list[:,0])))
    # print("TOTAL MEAN MAPE={}".format(np.mean(mae_list[:,1])))
    # print("TOTAL MEAN RMSE={}".format(np.mean(mae_list[:,2])))
    # t2 = time.time()
    # print("Total time spent: {:.4f}".format(t2 - t1))


    # ########### 自动进行单因子实验
    # # factor_index = [0]
    # factor_index = [0, 1, 2, 3, 6, 8]
    # for fac in factor_index:
    #     Config.fac_index = fac
    #     args.data = f'data/water/{place}/singleFac/{Config.fac_index}'
    #     run_once()
    #
    #


    # ########### 自动进行单因子实验,没个做5次，取平均值
    # t1 = time.time()
    # result_log = ""
    #
    #
    # factor_index = [0, 1, 2, 3, 6, 8]
    #
    # for fac in factor_index:
    #     args.data = f'data/water/{place}/singleFac/' + str(fac)
    #
    #     mae_list = []
    #     for i in range(5):
    #         print("data::{},running {} st".format(args.data,i+1))
    #         mae = run_once()
    #         print(mae)
    #         mae_list.append(mae)
    #     mae_list = np.array(mae_list)
    #     result_log += "data=={},MAE={:.4f},MAPE={:.4f},RMSE={:.4f}\n"\
    #         .format(args.data,np.mean(mae_list[:,0]),np.mean(mae_list[:,1]),np.mean(mae_list[:,2]) )
    #     print(result_log)
    #
    #
    # print('------------------finished-----------------------')
    # t2 = time.time()
    # print(result_log)
    # print("Total time spent: {:.4f}".format(t2 - t1))

