import water.SVR.data_generate as data_generate
import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from scipy.stats import pearsonr
import os
from sklearn.ensemble import BaggingRegressor

factors = ['pH值', '总氮', '总磷', '氨氮', '水温', '浑浊度', '溶解氧', '电导率', '高锰酸盐指数']


def run_one_fac(fac_index):
    x_train, y_train, x_val, y_val, x_test, y_test = \
        data_generate.load_single_data(fac_index=fac_index, y_bool=False, y_length=3)

    # maes=[]
    # 每个步长分别预测

    for step in range(3):
        # print(f'running {factors[fac_index]} for {step}th step')
        y_train_t = y_train[:,step]
        y_test_t = y_test[:,step]

        bag_svr = BaggingRegressor(base_estimator=SVR(), n_estimators=32, random_state=0)
        regr = make_pipeline(StandardScaler(), bag_svr)
        regr.fit(x_train, y_train_t)
        y_pred = regr.predict(x_test)

        # 保存预测值
        save_root = 'data/output/shangban/y/BagSVR'
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        np.savez_compressed(
            os.path.join(save_root, f"BagSVR_fac{fac_index}_step{step}.npz"),
            y_pred=y_pred,
            y_test=y_test_t
        )

        mae = metrics.mean_absolute_error(y_test_t, y_pred)
        rmse = metrics.mean_squared_error(y_test_t, y_pred) ** 0.5
        r2 = metrics.r2_score(y_test_t, y_pred)
        mape = metrics.mean_absolute_percentage_error(y_test_t, y_pred)
        r = pearsonr(y_test_t, y_pred)[0]
        print(f'{factors[fac_index]}{step}th,{mae:.4f},{rmse:.4f},{r2:.4f},{mape:.4f},{r:.4f}')


# 使用SVR预测水质
if __name__ == '__main__':
    # 循环预测所有因子
    print('MAE,RMSE,r2,MAPE,r')
    for fac in [0,1,2,3,6,8]:
        run_one_fac(fac)

    pass