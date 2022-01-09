import pandas as pd
import os
import numpy as np

## 生成上坂的高锰酸盐数据集

ids_shangban = [ '天宝大水港排涝站','中排渠涝站（天宝）',
              '甘棠溪慧民花园监测点',
        '康山溪金峰花园监测点', '芗城水利局站','中山桥水闸站', '北京路水闸站','九湖监测点','桂林排涝站','上坂']


# 生成数据集并保存
def generate(y_bool=True,post_fix=''):
    seq_length_x = 24
    seq_length_y = 3
    # 输出目录
    out_dir = 'data/dataset'
    # 门限值，超过即为异常数据
    gate = 10

    # 只读取高锰酸盐指数的数据
    df = pd.read_csv('data/water_4H.csv', usecols=[1, 2, 11])

    # 将每个站点的数据单独放一列
    merge = None
    for site in ids_shangban:
        one = df.loc[df['站点名称'] == site]
        one.columns = ['time', 'site', site]
        one = one[['time', site]]
        if merge is None:
            merge = one
        else:
            merge = pd.merge(merge, one, on='time')
    merge.set_index(keys='time', inplace=True)

    # 滑动窗口生成数据集
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(1, (seq_length_y + 1), 1))
    df = merge
    num_samples, num_nodes = df.shape
    data = df.values
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    print("x shape: ", x.shape, ", y shape: ", y.shape)

    # 标签值改为0、1
    if y_bool:
        y = np.where(y > gate, 1, 0)
        print(f'异常数据占比：{np.mean(y)}')

    # 将不同站点的数据拼接在一起。
    site_num = x.shape[2]
    x_concate = []
    y_concate = []
    for i in range(site_num):
        x_concate.append(x[:, :, i])
        y_concate.append(y[:, :, i])
    x = np.concatenate(x_concate)
    y = np.concatenate(y_concate)

    # 随机打乱！
    per = np.random.permutation(x.shape[0])
    x = x[per]
    y = y[per]

    # 数据集切分
    # Write the data into npz file.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(out_dir, f"{cat}_{post_fix}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


# 加载数据集
def load_data(post_fix=''):
    dataset_dir = 'data/dataset'
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, f"{category}_{post_fix}.npz"))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    return data['x_train'],data['y_train'],data['x_val'],data['y_val'],data['x_test'],data['y_test']


# 加载单因子数据集
def load_single_data(place='shangban',fac_index=0,y_bool=False,y_length=1):
    gate = 10

    dataset_dir = f'data/water/{place}/singleFac/{fac_index}'
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, f"{category}.npz"))
        # 去掉时间维
        data['x_' + category] = cat_data['x'][..., 0]
        data['y_' + category] = cat_data['y'][..., 0][:,0:y_length]

        # 将不同站点的数据拼接在一起。
        site_num = data['x_' + category].shape[2]
        x_concate = []
        y_concate = []
        for i in range(site_num):
            x_concate.append(data['x_' + category][:, :, i])
            y_concate.append(data['y_' + category][..., i])
        data['x_' + category] = np.concatenate(x_concate)
        data['y_' + category] = np.concatenate(y_concate)

        # 预警分类？
        if y_bool:
            data['y_' + category] = np.where(data['y_' + category] > gate, 1, 0)
            # print(f'异常数据占比：{np.mean(data["y_" + category])}')
    return data['x_train'], data['y_train'], data['x_val'], data['y_val'], data['x_test'], data['y_test']


# 合并站点数据，站点维度下标为1
def merge_site(data):
    site_num = data.shape[1]
    concate = []
    for i in range(site_num):
        concate.append(data[:, i])
    return np.concatenate(concate)


if __name__ == '__main__':
    # generate(False,'real')
    pass

    ## 运行结果
#x shape:  (1804, 24, 10) , y shape:  (1804, 3, 10)
# train x:  (1263, 24, 10) y: (1263, 3, 10)
# val x:  (180, 24, 10) y: (180, 3, 10)
# test x:  (361, 24, 10) y: (361, 3, 10)

