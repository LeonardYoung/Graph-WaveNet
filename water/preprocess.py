##
import pandas as pd
import time,datetime
import numpy as np
import os


# 预处理1，合并文件
# file_list: 文件列表
def data_fix_concat(root,file_type, dist):
    file_list = [os.path.join(path, name) for path, subdirs, files in os.walk(root) for name in files]
    df_list = []
    for file in file_list:
        print('处理文件：' + file)
        if file_type == 'csv':
            file1_df = pd.read_csv(file, usecols=[1, 2, 3, 4, 5, 6], dtype=object)
        else:
            file1_df = pd.read_excel(file, usecols=[1, 2, 3, 4, 5, 6], dtype=object)
        # print(file1_df.columns)

        # file1_df[file1_df['站点名称'] == '龙文上坂'] = '上坂'
        # file1_df.loc[file1_df['因子代码'] == 'W01001', '因子名称'] = 'pH值'

        if '因子代码' in file1_df.columns:
            file1_df.loc[file1_df['因子代码'] == 'W01001', '因子名称'] = 'pH值'  # “pH”替换为“pH值”
            file1_df.loc[file1_df['因子代码'] == 'W01003', '因子名称'] = '浑浊度'  # “浊度”替换为“浑浊度”
            file1_df.loc[file1_df['因子名称'] == '浊度', '因子名称'] = '浑浊度'

        if '站点编码' in file1_df.columns:
            file1_df.loc[file1_df['站点编码'] == '3506000005WQ', '站点名称'] = '芗城水利局站'  # 将“芗城区水利局站”和“舟尾亭水闸”替换为“芗城水利局站”
            file1_df.loc[file1_df['站点编码'] == '3506000002WQ', '站点名称'] = '北京路水闸站'  # 规范“北京路水闸”为“北京路水闸站”
            file1_df.loc[file1_df['站点编码'] == '3506000004WQ', '站点名称'] = '中山桥水闸站'  # 规范“中山桥水闸”为“中山桥水闸站”
            file1_df.loc[file1_df['站点编码'] == 'A350600_2009', '站点名称'] = '上坂'  # 规范“中山桥水闸”为“中山桥水闸站”
        # 删除两条无关数据
        if '因子代码' in file1_df.columns:
            file1_df = file1_df[~file1_df['因子代码'].isin(['W01017'])]
            file1_df = file1_df[~file1_df['因子代码'].isin(['W01018'])]

        file1_df.drop_duplicates(inplace=True)  # 删除重复数据
        df_list.append(file1_df)
    file_df = pd.concat(df_list, ignore_index=True)
    file_df.to_csv(dist, header=True, index=False, encoding='utf_8_sig')


# 预处理2 转置数据
def data_transpose(file_in='./first.csv', dataTrans="./second.csv"):
    print('转置数据...')
    data_df = pd.read_csv(file_in, encoding='utf-8', dtype=object)
    data_df.drop(['站点编码', '监测因子编码'], inplace=True, axis=1, errors='ignore')  # 删除“站点编码”、“监测因子编码”两个属性
    # 转置“因子名称”、“数值”两个属性
    data_df['监测时间'] = pd.to_datetime(data_df['监测时间'])
    data_df['数值'] = data_df['数值'].apply(pd.to_numeric, errors='coerce')
    data_df = data_df.pivot_table(index=['监测时间', '站点名称'], columns='因子名称', values='数值')
    data_df.reset_index(inplace=True)
    data_df.to_csv(dataTrans, header=True, index=False, encoding='utf_8_sig')
    print('转置数据完成...')


# 预处理3 插入缺失时间
def data_insert_time(file_in, file_out,start,end,freq):
    data2020_df = pd.read_csv(file_in, encoding='utf-8', dtype=object,
                              usecols=[0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # data2020_df.head(5)
    site_list = data2020_df['站点名称'].unique().tolist()

    all_df = pd.DataFrame()
    first = True

    date_index = pd.date_range(start,end,freq=freq)
    for site in site_list:
        site_df = data2020_df.loc[data2020_df['站点名称'] == site]
        for i in range(len(site_df)):
            row_time = datetime.datetime.strptime(site_df.iloc[i, 0], '%Y-%m-%d %H:%M:%S')
            site_df.iloc[i, 0] = row_time
        site_df = site_df.set_index('监测时间')
        site_df = site_df.reindex(date_index)
        site_df['站点名称'][site_df['站点名称'].isna()] = site

        all_df = all_df.append(site_df) if not first else site_df
        first = False

    all_df.to_csv(file_out,index_label='监测时间', header=True, index=True, encoding='utf_8_sig')


# 预处理4 数据缺失处理，异常处理
def data_clean(file_in, file_out, flag_save=False, fill_method = 'nearest'):
    print('数据缺失处理...')
    # 缺失值、异常值处理
    data_df = pd.read_csv(file_in, encoding='utf-8', dtype=object)
    data_df['phFlag'] = 0
    data_df['TNFlag'] = 0
    data_df['TPFlag'] = 0
    data_df['NHFlag'] = 0
    data_df['temperFlag'] = 0
    data_df['turbiFlag'] = 0
    data_df['doxygenFlag'] = 0
    data_df['conductFlag'] = 0
    data_df['permangaFlag'] = 0
    data_df.sort_values(by=['站点名称', '监测时间'], inplace=True)
    data_df['pH值'] = data_df['pH值'].astype(float)
    data_df['总氮'] = data_df['总氮'].astype(float)
    data_df['总磷'] = data_df['总磷'].astype(float)
    data_df['氨氮'] = data_df['氨氮'].astype(float)
    data_df['水温'] = data_df['水温'].astype(float)
    data_df['浑浊度'] = data_df['浑浊度'].astype(float)
    data_df['溶解氧'] = data_df['溶解氧'].astype(float)
    data_df['电导率'] = data_df['电导率'].astype(float)
    data_df['高锰酸盐指数'] = data_df['高锰酸盐指数'].astype(float)



    data_df['pH值'] = np.where(data_df['pH值'] > 50, data_df['pH值'] * 0.1, data_df['pH值'])  # 50<pH值<90：小数点左移一位
    data_df['pH值'][data_df['pH值'] > 14] = None  # pH值>14：视为缺失值，和缺失值一起处理
    data_df['pH值'][data_df['pH值'] < 1] = None  # pH值<1：视为缺失值，和缺失值一起处理
    data_df['phFlag'][data_df['pH值'].isna()] = 1
    data_df['pH值'].interpolate(method=fill_method, inplace=True) # 用前一时刻的值填充缺失值

    data_df['水温'][data_df['水温'] == 0] = None  # 水温=0：视为缺失值，和缺失值一起处理
    data_df['temperFlag'][data_df['水温'].isna()] = 1
    data_df['水温'].interpolate(method=fill_method, inplace=True)  # 用前一时刻的值填充缺失值

    data_df['turbiFlag'][data_df['浑浊度'].isna()] = 1
    data_df['浑浊度'][data_df['浑浊度'] == 0] = None
    data_df['浑浊度'].interpolate(method=fill_method, inplace=True)  # 用前一时刻的值填充缺失值

    data_df['溶解氧'][data_df['溶解氧'] == 0] = None
    data_df['溶解氧'][data_df['溶解氧'] >= 14.64] = None  # 溶解氧>14.64：视为缺失值，和缺失值一起处理
    data_df['doxygenFlag'][data_df['溶解氧'].isna()] = 1
    data_df['溶解氧'].interpolate(method=fill_method, inplace=True)  # 用前一时刻的值填充缺失值

    data_df['conductFlag'][data_df['电导率'].isna()] = 1
    data_df['电导率'][data_df['电导率'] == 0] = None
    data_df['电导率'].interpolate(method=fill_method, inplace=True)  # 用前一时刻的值填充缺失值

    # data_df.reset_index(inplace=True)
    data_df['总氮'][data_df['总氮'] <= 0] = None  # 总氮=0：视为缺失值，和缺失值一起处理
    data_df['总氮'][data_df['总氮'] > 100] = None  # 总氮>100：视为缺失值，和缺失值一起处理
    data_df['TNFlag'][data_df['总氮'].isna()] = 1
    data_df['总氮'].interpolate(method=fill_method, inplace=True)  # 用前一时刻的值填充缺失值

    data_df['氨氮'] = np.where(data_df['氨氮'] < 0, data_df['氨氮'] * -1, data_df['氨氮'])  # 负数转正
    data_df['氨氮'][data_df['氨氮'] == 0] = None  # 氨氮=0：视为缺失值，和缺失值一起处理
    # data_df['氨氮'] = np.where(data_df['氨氮'] > data_df['总氮'], None, data_df['氨氮'])
    data_df['NHFlag'][data_df['氨氮'].isna()] = 1
    data_df['氨氮'].interpolate(method=fill_method, inplace=True)  # 用前一时刻的值填充缺失值

    data_df['总磷'] = np.where(data_df['总磷'] < 0, data_df['总磷'] * -1, data_df['总磷'])  # 负数转正
    data_df['总磷'][data_df['总磷'] == 0] = None  # 总磷=0：视为缺失值，和缺失值一起处理
    data_df['总磷'] = np.where(data_df['总磷'] > 5, data_df['总磷'] * 0.1, data_df['总磷'])  # 总磷>5：小数点左移一位
    data_df['TPFlag'][data_df['总磷'].isna()] = 1
    data_df['总磷'].interpolate(method=fill_method, inplace=True)  # 用前一时刻的值填充缺失值

    data_df['高锰酸盐指数'] = np.where(data_df['高锰酸盐指数'] < 0, data_df['高锰酸盐指数'] * -1,
                                    data_df['高锰酸盐指数'])  # 负数转正
    data_df['高锰酸盐指数'][data_df['高锰酸盐指数'] == 0] = None  # 高锰酸盐指数=0：视为缺失值，和缺失值一起处理
    data_df['permangaFlag'][data_df['高锰酸盐指数'].isna()] = 1
    data_df['高锰酸盐指数'].interpolate(method=fill_method, inplace=True)  # 用前一时刻的值填充缺失值
    data_df.reset_index(inplace=True)

    if flag_save:
        data_df.to_csv(file_out, header=True, index=False, encoding='utf_8_sig')
    else:
        data_df.to_csv(file_out, header=True, index=False, encoding='utf_8_sig')
    print('数据缺失处理完成')


# 统计每个站点每个因子数据缺失率，
# 并按照（,pH值,总氮,总磷,氨氮,水温,浑浊度,溶解氧,电导率,高锰酸盐指数）顺序打印出来
def loss_data_count(data_file):
    chang_df = pd.read_csv(data_file)

    sites = chang_df['站点名称'].unique()

    flags = ['phFlag', 'TNFlag', 'TPFlag', 'NHFlag', 'temperFlag'
        , 'turbiFlag', 'doxygenFlag', 'conductFlag', 'permangaFlag']

    loss_count = 0
    print('站点,pH值,总氮,总磷,氨氮,水温,浑浊度,溶解氧,电导率,高锰酸盐指数')
    for site in sites:
        site_len = len(chang_df[chang_df['站点名称'] == site])

        line = "{}".format(site)
        for flag in flags:
            temp_df = chang_df[chang_df[flag] == 1]
            temp_df = temp_df[temp_df['站点名称'] == site]
            line += str(",{:.3f}".format(len(temp_df) / site_len))
            loss_count += len(temp_df)
        print(line)

    line = "全站点"
    for flag in flags:
        temp_df = chang_df[chang_df[flag] == 1]
        line += str(",{:.3f}".format(len(temp_df) / len(chang_df)))
    print(line)


if __name__ == '__main__':
    # # 预处理监测站点数据
    # ######## 上坂
    # # 步骤1：
    # root = r"E:\project\mvp\Graph-WaveNet\data\water\shangban\origin"
    # data_fix_concat(root,'xls','./temp/1.csv')
    #
    # # 步骤2：
    # data_transpose('./temp/1.csv', './temp/2.csv')
    # # 步骤3：
    # data_insert_time('./temp/2.csv','./temp/3.csv','2020-01-01 00:00:00', '2020-10-31 23:00:00','4H')
    # 步骤4
    data_clean('./temp/3.csv', './temp/4.csv',False,'quadratic')

    # # ######### 长泰
    # # 步骤1：
    # root = r"E:\project\mvp\Graph-WaveNet\data\water\changtai\origin"
    # data_fix_concat(root, 'csv', './temp/a.csv')
    #
    # # 步骤2：
    # data_transpose('./temp/a.csv', './temp/b.csv')
    # # 步骤3：
    # data_insert_time('./temp/b.csv', './temp/c.csv', '2019-10-01 00:00:00', '2021-09-30 23:00:00', '4H')
    # # 步骤4
    # data_clean('./temp/c.csv', './temp/d.csv')



