import numpy as np
from water.SVR.data_generate import merge_site


# 加载保存的模型预测值和真实值，仅限于浅层模型
def load_shallow(type='BagSVR',place='shangban'):
    all_test_list = []
    all_pred_list = []
    for fac in [0, 1, 2, 3, 6, 8]:
        y_test_list = []
        y_pred_list = []
        for step in range(3):
            y = np.load(f'data/output/{place}/y/{type}/{type}_fac{fac}_step{step}.npz')

            y_test_list.append(y['y_test'])
            y_pred_list.append(y['y_pred'])

        all_test_list.append(np.stack(y_test_list))
        all_pred_list.append(np.stack(y_pred_list))
    all_test_list = np.stack(all_test_list)
    all_pred_list = np.stack(all_pred_list)
    return all_test_list,all_pred_list


# 加载保存的模型预测值和真实值，仅限于WaveNet模型
def load_waveNet(model='GCNLSTM',place='shangban'):
    # model_dir = 'GCNLSTM'
    # model_dir = 'noGCNLSTM'
    # model_dir = 'GCNnoLSTM'
    # model_dir = 'noGCNnoLSTM'

    all_test_list = []
    all_pred_list = []
    for fac in [0, 1, 2, 3, 6, 8]:
        y = np.load(f'data/output/{place}/y/singleWaveNet/{model}/{fac}/out.npz')
        y_test = merge_site(y['y_test'])
        y_pred = merge_site(y['y_pred'])

        all_test_list.append(np.transpose(y_test, (1, 0)))
        all_pred_list.append(np.transpose(y_pred, (1, 0)))

        # print(y_test.transpose(0,1).shape)
    all_test_list = np.stack(all_test_list)
    all_pred_list = np.stack(all_pred_list)
    return all_test_list, all_pred_list


# 加载保存的所有模型预测值和真实值，返回两个np的shape都是(5, 6, 3, 3610)，分别为模型维，因子维，步长维，数据维
def load_all_y(place='shangban'):
    all_test_list = []
    all_pred_list = []
    svr_y_test, svr_y_pred = load_shallow(place=place)

    all_test_list.append(svr_y_test)
    all_pred_list.append(svr_y_pred)

    model_dirs = ['noGCNnoLSTM', 'noGCNLSTM', 'GCNnoLSTM', 'GCNLSTM']

    for model in model_dirs:
        y_test, y_pred = load_waveNet(model,place=place)
        all_test_list.append(y_test)
        all_pred_list.append(y_pred)

    all_test_list = np.stack(all_test_list)
    all_pred_list = np.stack(all_pred_list)
    return all_test_list, all_pred_list



