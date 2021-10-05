import pandas as pd
import numpy as np
import os

# 按照上下游顺序
ids = [ '天宝大水港排涝站','中排渠涝站（天宝）',
             '程溪下庄工业区水质监测点', '甘棠溪慧民花园监测点',
        '康山溪金峰花园监测点', '芗城水利局站','中山桥水闸站', '北京路水闸站','九湖监测点','桂林排涝站','上坂']

# 表格中的顺序
factors = ['pH值', '总氮', '总磷', '氨氮', '水温', '浑浊度', '溶解氧', '电导率', '高锰酸盐指数']


# 横向合并一个因子，保存为h5
def merge_one_factor(inc = 0):
    df = pd.read_csv('../data/water/water2020.csv', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    merge = None
    for site in ids:
        one = df.loc[df['站点名称'] == site]
        one.columns = ['time', 'site'] + [site + str(i) for i in range(9)]
        one = one[['time'] + [site + str(inc)]]
        if merge is None:
            merge = one
        else:
            merge = pd.merge(merge, one, on='time')
    merge.set_index(keys='time', inplace=True)
    file_name = '../data/water/single/merge' + str(inc) +'.h5'
    merge.to_hdf(file_name, key='merge', index=False)
    return file_name


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    feature_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values.astype("datetime64") - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)
    if add_day_in_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    data = np.concatenate(feature_list, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_one_factor(hdf_file, out_dir,seq_length_x=24, seq_length_y=24):
    # seq_length_x, seq_length_y = 24, 24
    df = pd.read_hdf(hdf_file)
    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, (seq_length_y + 1), 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
    )
    print("x shape: ", x.shape, ", y shape: ", y.shape)
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

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(out_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def generate_one_site_one_factor(factor_index,site_index,
                                 seq_length_x=24, seq_length_y=3):
    hdf_file = '../data/water/single/merge' + str(factor_index) +'.h5'
    site_names = "abcdefghijklmnopq"
    out_dir = '../data/water/singlesingle/{}{}'.format(factor_index,site_names[site_index])

    df = pd.read_hdf(hdf_file)
    df = df.iloc[:,site_index]
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(1, (seq_length_y + 1), 1))

    num_samples = len(df)
    data = np.expand_dims(df.values, axis=-1)

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    x = np.squeeze(x)
    y = np.squeeze(y)

    print("x shape: ", x.shape, ", y shape: ", y.shape)
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

    exi =  os.path.exists(out_dir)
    if not exi:
        os.mkdir(out_dir)

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)

        np.savez_compressed(
            os.path.join(out_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )



# 生成邻接矩阵的文件
def get_adj_file(num_nodes,file_name):

    id_to_inc = {}
    for i in range(len(ids)):
        id_to_inc[ids[i]] = i
    # id_to_inc = {
    #     '上坂': 0,
    #     '中山桥水闸站': 1,
    #     '中排渠涝站（天宝）': 2,
    #     '九湖监测点': 3,
    #     '北京路水闸站': 4,
    #     '天宝大水港排涝站': 5,
    #     '康山溪金峰花园监测点': 6,
    #     '桂林排涝站': 7,
    #     '甘棠溪慧民花园监测点': 8,
    #     '程溪下庄工业区水质监测点': 9,
    #     '芗城水利局站': 10,
    # }
    adj = np.ones([num_nodes, num_nodes])
    pickle_data = [ids, id_to_inc, adj]
    import pickle
    with open('../data/water/adjs/' + file_name, "wb") as myprofile:
        pickle.dump(pickle_data, myprofile)


# 1.标准化
def norm_data(input_csv, output):
    data_df = pd.read_csv(input_csv, encoding='utf-8', dtype=object)
    columns = data_df.columns.tolist()
    norm_data_df = pd.DataFrame()

    # 9个维度都做标准化
    norm_data_df['监测时间'] = data_df['监测时间']
    norm_data_df['站点名称'] = data_df['站点名称']
    for i in range(9):
        col = columns[i + 3]

        data_df[col] = data_df[col].astype(float)
        # mean = data_df[col].mean()
        # std = data_df[col].std()
        # norm = (data_df[col] - mean) / std
        min = data_df[col].min()
        max = data_df[col].max()
        norm = (data_df[col] - min) / (max - min)
        # print('%s: mean:%f  std:%f   ' % (col, mean, std))
        norm_data_df[col] = norm

    norm_data_df.to_csv(output, index=False)


# 2.站点分离
def merge_all_factor(input_csv, output):
    df = pd.read_csv(input_csv, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11])
    merge = None
    for site in df['站点名称'].unique().tolist():
        one = df.loc[df['站点名称'] == site]
        one.columns = ['time', 'site'] + [site + str(i) for i in range(9)]
        one = one[['time'] + [site + str(i) for i in range(9)]]
        if merge is None:
            merge = one
        else:
            merge = pd.merge(merge, one, on='time')
    merge.set_index(keys='time', inplace=True)
    merge.to_hdf(output, key='merge', index=False)


# 3.生成数据
def generate_data(input_csv, output_dir):
    df = pd.read_hdf(input_csv)
    args = {}

    seq_length_x, seq_length_y, y_start = 24, 24, 1
    site_num, factor_num = 11, 9

    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(y_start, (seq_length_y + 1), 1))

    num_samples, num_nodes = df.shape

    data_site_list = []
    for site in range(site_num):
        start_ind = site * factor_num
        data_site = df.values[:, start_ind:start_ind + factor_num]
        data_site = np.expand_dims(data_site, axis=-1)
        data_site_list.append(data_site)
    data = np.concatenate(data_site_list, axis=-1)
    data = data.transpose((0, 2, 1))

    feature_list = [data]
    time_ind = (df.index.values.astype("datetime64") - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    time_in_day = np.tile(time_ind, [1, site_num, 1]).transpose((2, 1, 0))
    feature_list.append(time_in_day)

    data = np.concatenate(feature_list, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    print("x shape: ", x.shape, ", y shape: ", y.shape)
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

    # output_dir = 'data/water/normGen'
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


if __name__ == "__main__":
    # norm_data('../data/water/water2020.csv', '../data/water/norm2020.csv')
    # merge_all_factor('../data/water/norm2020.csv', '../data/water/normMergeAll.h5')
    # generate_data('../data/water/normMergeAll.h5', '../data/water/normGen')

    # merge_all_factor('../data/water/water2020.csv', '../data/water/mergeAll.h5')
    # generate_data('../data/water/mergeAll.h5', '../data/water/genAll')

    # # 生成单因子数据集
    # for i in range(9):
    #     file_name = merge_one_factor(i)
    #     generate_one_factor(file_name,'../data/water/single/'+str(i)+'/',24,24)

    # 全站点单因子
    # file_name = merge_one_factor(0)
    # generate_one_factor(file_name, '../data/water/single/' + str(0) + '/', 12, 3)

    # 单站点单因子
    for site in range(11):
        generate_one_site_one_factor(0,site,24,9)

    # get_adj_file(11,'adjOnes.pkl')