import numpy as np
from water.SVR.data_generate import merge_site

# 所有因子
factors = ['pH值', '总氮', '总磷', '氨氮', '水温', '浑浊度', '溶解氧', '电导率', '高锰酸盐指数']
factors_en = ['pH', 'TN', 'TP', 'NH$_3$', '水温', '浑浊度', 'DO', '电导率', 'CODmn']

# 有使用的因子
factors_use_en = ['pH', 'TN', 'TP', 'NH$_3$', 'DO', 'CODmn']
factors_use = ['pH值', '总氮', '总磷', '氨氮', '溶解氧', '高锰酸盐指数']

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


# 返回目前的数据结果名称，用于设置保存的目录名称
def model_names():
    return ['bagSVR','RF','noGCNnoLSTM', 'noGCNLSTM', 'GCNnoLSTM', 'GCNLSTM']

# 返回目前的数据结果名称，用于图标标题、论文
def model_full_names():
    return ['SVR','RF','WaveNet', 'WaveNet-LSTM', 'WaveNet-MGCN', 'WaveNet-LSTM-MGCN']

# 加载保存的所有模型预测值和真实值，返回两个np的shape都是(5, 6, 3, 3610)，分别为模型维，因子维，步长维，数据维
def load_all_y(place='shangban'):
    all_test_list = []
    all_pred_list = []

    svr_y_test, svr_y_pred = load_shallow(place=place)
    all_test_list.append(svr_y_test)
    all_pred_list.append(svr_y_pred)

    rf_y_test, rf_y_pred = load_shallow(type='RF',place=place)
    all_test_list.append(rf_y_test)
    all_pred_list.append(rf_y_pred)

    model_dirs = ['noGCNnoLSTM', 'noGCNLSTM', 'GCNnoLSTM', 'GCNLSTM']

    for model in model_dirs:
        y_test, y_pred = load_waveNet(model,place=place)
        all_test_list.append(y_test)
        all_pred_list.append(y_pred)

    all_test_list = np.stack(all_test_list)
    all_pred_list = np.stack(all_pred_list)
    return all_test_list, all_pred_list


def np_rmspe(y_true, y_pred):
    loss = np.sqrt(np.mean(np.square(((y_true - y_pred) / (y_true + np.mean(y_true)))), axis=0))
    return loss


# 计算mape，公式经过更改。每次除以标签值加上所有标签的均值，最后结果乘以2
def np_mape(y_true, y_pred):
    loss = np.abs(y_true - y_pred) / (y_true + np.mean(y_true))
    return np.mean(loss) * 2


# 打印出指定模型的所有metric
def model_metric(model_idx = 0):
    # import water.common as water_common
    # from water.common import np_rmspe
    from sklearn import metrics
    from scipy.stats import pearsonr
    factors_use_en = ['pH', 'TN', 'TP', 'NH$_3$', 'DO', 'CODmn']
    factors_use = ['pH值', '总氮', '总磷', '氨氮', '溶解氧', '高锰酸盐指数']

    all_test, all_pred = load_all_y()

    mae_list = []
    mape_list = []
    rmse_list = []
    rmspe_list = []
    r2_list = []
    r_list = []

    # 遍历每个因子
    for fac_idx in range(all_test.shape[1]):
        for step in range(3):
            y_test_t = all_test[model_idx, fac_idx, step, :]
            y_pred_t = all_pred[model_idx, fac_idx, step, :]

            mae = metrics.mean_absolute_error(y_test_t, y_pred_t)
            # mape = metrics.mean_absolute_percentage_error(y_test_t, y_pred_t)
            mape = np_mape(y_test_t,y_pred_t)
            rmse = metrics.mean_squared_error(y_test_t, y_pred_t) ** 0.5
            rmspe = np_rmspe(y_test_t, y_pred_t)
            # rmspe2 = masked_rmspe(y_test_t,y_pred_t)
            r2 = metrics.r2_score(y_test_t, y_pred_t)
            r = pearsonr(y_test_t, y_pred_t)[0]

            print(f'{factors_use[fac_idx]}{step}th,{mae:.3f},{mape:.3f},'
                  f'{rmse:.3f},{rmspe:.3f},{r2:.3f},{r:.3f}')
            # break
            # break

            mae_list.append(mae)
            mape_list.append(mape)
            rmse_list.append(rmse)
            rmspe_list.append(rmspe)
            r_list.append(r)
            r2_list.append(r2)

    # 计算平均值
    print(f'平均,{np.mean(mae_list):.3f},{np.mean(mape_list):.3f},'
          f'{np.mean(rmse_list):.3f},{np.mean(rmspe_list):.3f},'
          f'{np.mean(r2_list):.3f},{np.mean(r_list):.3f}')


# 打印出上坂每个因子的统计数据
def print_statics():
    import pandas as pd
    df = pd.read_csv('water/temp/4.csv')
    print(df.columns)
    print('min,max,mean,std,skew,kurt')
    for col in df.columns[3:12]:
        print(f'{col},{df[col].min():.3f},{df[col].max():.3f},{df[col].mean():.3f}'
              f',{df[col].std():.3f},{df[col].skew():.3f},{df[col].kurt():.3f}')


if __name__ == '__main__':
    model_metric(2)