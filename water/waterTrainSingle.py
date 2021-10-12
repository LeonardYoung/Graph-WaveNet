import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer
from utils import earlystopping


parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:1',help='')
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
parser.add_argument('--save',type=str,default='./garage/metr',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

args = parser.parse_args()


def train(engine,dataloader):
    dataloader['train_loader'].shuffle()
    for step, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
        train_x = torch.Tensor(x).to(engine.device).transpose(1, 3)
        train_y = torch.Tensor(y).to(engine.device).transpose(1, 3)
        engine.train(train_x, train_y[:, 0, :, :])


def validate(engine,dataloader):
    for step, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
        test_x = torch.Tensor(x).to(engine.device).transpose(1, 3)
        test_y = torch.Tensor(y).to(engine.device).transpose(1, 3)
        metrics = engine.eval(test_x, test_y[:, 0, :, :])
        return metrics


def test(engine,dataloader,model_path):

    engine.model.load_state_dict(torch.load(model_path))
    scaler = dataloader['scaler']

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(engine.device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

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

    amae = []
    amape = []
    armse = []
    for i in range(args.seq_length):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))


def main():
    #set seed
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)

    #load data
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    model_save_path = "./data/save_models/singFactor/waveNet.pth"
    early_stopping = earlystopping.EarlyStopping(patience=20, path=model_save_path, verbose=True)

    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit)

    print("start training...",flush=True)

    for e in range(1,args.epochs+1):
        train(engine, dataloader)

        metrics =validate(engine, dataloader)
        print("Epoch:{},validate loss:{}".format(e, metrics[0]))

        early_stopping(metrics[0],engine.model)
        if early_stopping.early_stop:
            print("Early stopping.")
            break
    test(engine,dataloader,model_save_path)


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
