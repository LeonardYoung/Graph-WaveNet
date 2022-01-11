


#  绘制mape、rmspe柱状图
def chart_mape_rmspe():
    import water.common as water_common
    from sklearn import metrics
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    factors_use_en = ['pH', 'TN', 'TP', 'NH$_3$', 'DO', 'CODmn']
    factors_use = ['pH值', '总氮', '总磷', '氨氮', '溶解氧', '高锰酸盐指数']

    all_test, all_pred = water_common.load_all_y()
    # print(all_test.shape)
    model_num = all_test.shape[0]

    plt.rcParams.update({'font.size': 22})
    # 遍历每个因子
    for fac_idx in range(all_test.shape[1]):
        model_mape_list = []
        model_rmspe_list = []
        for model_idx in range(all_test.shape[0]):
            mape_mean = []
            rmspe_mean = []
            for step in range(3):
                y_test = all_test[model_idx, fac_idx, step, :]
                y_pred = all_pred[model_idx, fac_idx, step, :]

                mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
                rmspe = water_common.np_rmspe(y_test, y_pred)
                mape_mean.append(mape)
                rmspe_mean.append(rmspe)

            # 3个步长取平均
            mape_mean = np.mean(mape_mean)
            rmspe_mean = np.mean(rmspe_mean)

            model_mape_list.append(mape_mean)
            model_rmspe_list.append(rmspe_mean)

        # print(model_mape_list)
        # print(model_rmspe_list)
        model_mape_list = np.array(model_mape_list)
        model_rmspe_list = np.array(model_rmspe_list)

        xx = np.array(range(model_num))
        # print(xx)
        fig, ax = plt.subplots()
        # name_list = ['SVR', 'noGCNnoLSTM', 'noGCNLSTM', 'GCNnoLSTM', 'GCNLSTM']
        name_list = water_common.model_full_names()
        ax.bar(xx, model_mape_list, label='MAPE', width=0.4, tick_label=name_list)
        ax.bar(xx, -model_rmspe_list, label='rmspe', width=0.4)
        plt.title(factors_use_en[fac_idx])
        plt.xticks(rotation=45, ha='right')
        # plt.legend(loc='center left')
        # plt.legend()

        # 保存
        save_root = f'data/output/shangban/figure/merge'
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        # plt.title(f'{factors_use_en[fac_idx]}')

        plt.savefig(f'{save_root}/mape柱状图_{factors_use[fac_idx]}.png'
                    , dpi=960, bbox_inches='tight')

        plt.show()

        # break


# 绘制loss曲线
def chart_loss():
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    factors = ['pH值', '总氮', '总磷', '氨氮', '水温', '浑浊度', '溶解氧', '电导率', '高锰酸盐指数']
    factors_en = ['pH', 'TN', 'TP', 'NH$_3$', '水温', '浑浊度', 'DO', '电导率', 'CODmn']
    model_dir = 'GCNLSTM'
    # model_dir = 'noGCNLSTM'
    # model_dir = 'GCNnoLSTM'
    # model_dir = 'noGCNnoLSTM'
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams.update({'font.size': 22})

    for fac in [0, 1, 2, 3, 6, 8]:
        loss = np.load(f'data/output/shangban/loss/singleWaveNet/{model_dir}/{fac}/loss.npz')
        train_loss = loss['train_loss']
        val_loss = loss['val_loss']
        xx = np.array(range(train_loss.size))
        # 画图
        fig = plt.figure()
        ax = fig.add_subplot()
        # ax.set_aspect(1000)

        plt.xlabel('Epoch')
        plt.ylabel('Loss(MAE)')
        # plt.rcParams['figure.figsize']=(5,5)

        ax.plot(xx, train_loss, color='red', linewidth=2, label='train')
        ax.plot(xx, val_loss, color='blue', linewidth=2, label='validate')

        plt.legend()

        # 保存
        save_root = f'data/output/shangban/figure/{model_dir}'
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        plt.title(f'{factors_en[fac]}')

        plt.savefig(f'{save_root}/loss_{factors[fac]}.png'
                    , dpi=960, bbox_inches='tight')
        plt.show()
        # break


# 绘制回归直线和散点图
def chart_regression(model_idx,place='shangban'):
    import water.common as water_common
    from sklearn import linear_model
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    factors_use_en = ['pH', 'TN', 'TP', 'NH$_3$', 'DO', 'CODmn']
    factors_use = ['pH值', '总氮', '总磷', '氨氮', '溶解氧', '高锰酸盐指数']

    all_test, all_pred = water_common.load_all_y()
    model_name = water_common.model_names()[model_idx]
    model_full_name = water_common.model_full_names()[model_idx]
    plt.rcParams.update({'font.size': 22})
    # plt.rcParams['font.sans-serif']=['SimHei']#显示中文标签
    # 遍历每个因子
    for fac_idx in range(all_test.shape[1]):
        y_test = []
        y_pred = []
        for step in range(3):
            y_test_t = all_test[model_idx, fac_idx, step, :]
            y_pred_t = all_pred[model_idx, fac_idx, step, :]

            y_test.append(y_test_t)
            y_pred.append(y_pred_t)
        y_test = np.concatenate(y_test)
        y_pred = np.concatenate(y_pred)

        regr = linear_model.LinearRegression()
        # 拟合
        y_test = y_test.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        regr.fit(y_test, y_pred)  # 注意此处.reshape(-1, 1)，因为X是一维的！

        # 不难得到直线的斜率、截距
        k, b = regr.coef_, regr.intercept_
        r2 = regr.score(y_test, y_pred)

        # 画图
        fig = plt.figure()
        ax = fig.add_subplot()
        # 1.真实的点
        ax.scatter(y_test, y_pred, color='blue', linewidths=1)

        # 2.拟合的直线
        ax.plot(y_test, regr.predict(y_test), color='red', linewidth=2)

        plt.xlabel(f'Observed {factors_use_en[fac_idx]}')
        plt.ylabel(f'predicted {factors_use_en[fac_idx]}')
        plt.title(f'{model_full_name} for {factors_use_en[fac_idx]}', fontsize=20)

        # 写文字
        plt.text(0.05, 0.75, f'y={k[0][0]:.3f}x+{b[0]:.3f}\nR$^2$={r2:.3f}',
                 transform=ax.transAxes)

        # 保存
        save_root = f'data/output/{place}/figure/{model_name}'
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        plt.savefig(f'{save_root}/回归_{factors_use[fac_idx]}.png'
                    , dpi=960, bbox_inches='tight')
        plt.show()


# 绘制时间序列图，数据太多，只选取步长为1的前一百个数
def chart_timeseries(place='shangban'):
    import water.common as water_common
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    factors_use_en = ['pH', 'TN', 'TP', 'NH$_3$', 'DO', 'CODmn']
    factors_use = ['pH值', '总氮', '总磷', '氨氮', '溶解氧', '高锰酸盐指数']

    all_test, all_pred = water_common.load_all_y()
    model_full_name = water_common.model_full_names()

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['figure.figsize'] = (18, 6)
    plt.rcParams.update({'font.size': 26})
    seq_len = 100
    # 遍历每个因子
    for fac_idx in range(all_test.shape[1]):
        observed = False
        fig, ax = plt.subplots()
        plt.xlabel('time')
        plt.ylabel('value')

        for model_idx in range(all_test.shape[0]):
            for step in range(3):
                y_test = all_test[model_idx, fac_idx, step, :][:seq_len]
                y_pred = all_pred[model_idx, fac_idx, step, :][:seq_len]

                xx = np.array(range(y_test.size))
                if not observed:
                    ax.plot(xx, y_test, linewidth=2, label='Observed')
                    observed = True

                ax.plot(xx, y_pred, linewidth=2, label=model_full_name[model_idx])
                break

        plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
        # 保存
        save_root = f'data/output/{place}/figure/merge'
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        plt.title(f'{factors_use_en[fac_idx]} predict')
        plt.savefig(f'{save_root}/时间序列_{factors_use[fac_idx]}.png'
                    , dpi=960, bbox_inches='tight')
        plt.show()


# 绘制箱型图
def chart_box(place='shangban'):
    import water.common as water_common
    import os
    import matplotlib.pyplot as plt
    factors_use_en = ['pH', 'TN', 'TP', 'NH$_3$', 'DO', 'CODmn']
    factors_use = ['pH值', '总氮', '总磷', '氨氮', '溶解氧', '高锰酸盐指数']
    factors = ['pH值', '总氮', '总磷', '氨氮', '水温', '浑浊度', '溶解氧', '电导率', '高锰酸盐指数']
    factors_en = ['pH', 'TN', 'TP', 'NH$_3$', '水温', '浑浊度', 'DO', '电导率', 'CODmn']

    all_test, all_pred = water_common.load_all_y()
    model_full_name = water_common.model_full_names()

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['figure.figsize'] = (18, 6)
    plt.rcParams.update({'font.size': 26})
    seq_len = 100
    # 遍历每个因子
    for fac_idx in range(all_test.shape[1]):
        observed = False
        fig, ax = plt.subplots()
        values = []
        for model_idx in range(all_test.shape[0]):
            for step in range(3):
                y_test = all_test[model_idx, fac_idx, step, :][:seq_len]
                y_pred = all_pred[model_idx, fac_idx, step, :][:seq_len]

                # xx = np.array(range(y_test.size))
                if not observed:
                    values.append(y_test)
                    observed = True
                values.append(y_pred)

                # 只展示步长为1的数据
                break
        # 绘图
        plt.boxplot(x=values,
                    labels=['observed'] + model_full_name)

        plt.xticks(rotation=45, ha='right')

        # 保存
        save_root = f'data/output/shangban/figure/merge'
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        plt.title(f'box plot for {factors_use_en[fac_idx]} predict models')
        plt.savefig(f'{save_root}/箱型图_{factors_use[fac_idx]}.png'
                    , dpi=960, bbox_inches='tight')
        plt.show()



if __name__ == '__main__':

    # 绘制回归曲线
    for i in range(5):
        chart_regression(i)

