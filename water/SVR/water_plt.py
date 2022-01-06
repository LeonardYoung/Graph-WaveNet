import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

factors = ['pH值', '总氮', '总磷', '氨氮', '水温', '浑浊度', '溶解氧', '电导率', '高锰酸盐指数']


def plot_regresion():
    for fac in [0, 1, 2, 3, 6, 8]:
        for step in range(3):
            y = np.load(f'data/output/y/SVR/SVR_fac{fac}_step{step}.npz')
            # 建立线性回归模型
            regr = linear_model.LinearRegression()
            # 拟合
            y_test = y['y_test'].reshape(-1, 1)
            y_pred = y['y_pred'].reshape(-1, 1)
            regr.fit(y_test, y_pred)  # 注意此处.reshape(-1, 1)，因为X是一维的！

            # 不难得到直线的斜率、截距
            k, b = regr.coef_, regr.intercept_
            r2 = regr.score(y_test, y_pred)

            # 画图
            fig = plt.figure()
            ax = fig.add_subplot()
            # 1.真实的点
            ax.scatter(y_test, y_pred, color='blue')

            # 2.拟合的直线
            ax.plot(y_test, regr.predict(y_test), color='red', linewidth=2)

            plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
            # plt.rcParams[‘axes.unicode_minus’]=False
            plt.xlabel('真实值', fontsize=16)
            plt.ylabel('预测值', fontsize=16)
            plt.title(f'SVR预测{factors[fac]}（步长{step + 1}）', fontsize=16)

            plt.text(0.05, 0.8, f'y={k[0][0]:.3f}x+{b[0]:.3f}\nR2={r2:.3f}',
                     fontsize=16, transform=ax.transAxes)
            plt.savefig(f'data/output/figure/SVR/SVR{factors[fac]}_步长{step + 1}_回归.png'
                        , dpi=960)
            plt.show()

