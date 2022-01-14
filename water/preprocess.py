##
import pandas as pd
import time,datetime
import numpy as np
import os


ids_shangban = [ '天宝大水港排涝站','中排渠涝站（天宝）',
              '甘棠溪慧民花园监测点',
        '康山溪金峰花园监测点', '芗城水利局站','中山桥水闸站', '北京路水闸站','九湖监测点','桂林排涝站','上坂']

# 表格中的顺序
factors = ['pH值', '总氮', '总磷', '氨氮', '水温', '浑浊度', '溶解氧', '电导率', '高锰酸盐指数']
# 有用的因子
factors_use = ['pH值', '总氮', '总磷', '氨氮', '溶解氧', '高锰酸盐指数']


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

    # print(data_df.describe())


# 预处理3 插入缺失时间
def data_insert_time(file_in, file_out,start,end,freq):
    data2020_df = pd.read_csv(file_in, encoding='utf-8', dtype=object,
                              usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # data2020_df.head(5)
    site_list = data2020_df['站点名称'].unique().tolist()

    for fac in factors:
        data2020_df[fac] = data2020_df[fac].astype(float)
    all_df = pd.DataFrame()
    first = True

    date_index = pd.date_range(start, end, freq='1H')

    for site in site_list:
        site_df = data2020_df.loc[data2020_df['站点名称'] == site]
        site_df['监测时间'] = pd.to_datetime(site_df['监测时间'])
        site_df = site_df.set_index('监测时间')
        # 补全缺失时间
        site_df = site_df.reindex(date_index)
        # 按照间隔时间重新采样，多个数则取平均
        site_df = site_df.resample(freq).mean()
        site_df = site_df.round(3)
        site_df.insert(0, '站点名称', site)
        # site_df['站点名称'][site_df['站点名称'].isna()] = site

        all_df = all_df.append(site_df) if not first else site_df
        first = False

    all_df.to_csv(file_out, index_label='监测时间', header=True, index=True, encoding='utf_8_sig')


# 利用四分位间距检测异常值
def strange_data(df,col):

    # 计算超标数据个数和占比（三类）
    # out_df = df
    # if col == 'pH值':
    #     out_df = df[(df[col] < 6) | (df[col] > 9)]
    # elif col == '总氮':
    #     out_df = df[ df[col] > 1]
    # elif col == '总磷':
    #     out_df = df[ df[col] > 0.2]
    # elif col == '氨氮':
    #     out_df = df[ df[col] > 1]
    # elif col == '溶解氧':
    #     out_df = df[ df[col] < 5]
    # elif col == '高锰酸盐指数':
    #     out_df = df[ df[col] > 6]
    # else:
    #     return
    #
    # print('==={}超标点个数为：{},比例：{:.3f}%'.format(col, len(out_df),len(out_df)/len(df)*100))

    # 计算超标数据个数和占比（五类）
    # out_df = df
    # if col == 'pH值':
    #     out_df = df[(df[col] < 6) | (df[col] > 9)]
    # elif col == '总氮':
    #     out_df = df[ df[col] > 2]
    # elif col == '总磷':
    #     out_df = df[ df[col] > 0.4]
    # elif col == '氨氮':
    #     out_df = df[ df[col] > 2]
    # elif col == '溶解氧':
    #     out_df = df[ df[col] < 2]
    # elif col == '高锰酸盐指数':
    #     out_df = df[ df[col] > 15]
    # else:
    #     return

    # print('==={}超标点个数为：{},比例：{:.3f}%'.format(col, len(out_df),len(out_df)/len(df)*100))

    if col == '电导率':
        data_bottom,data_top = 125,750
    else:
        q_75 = df[col].quantile(q=0.75)
        q_25 = df[col].quantile(q=0.25)
        d = q_75 - q_25
        # 求数据上界和数据下界
        data_top = q_75 + 1.5 * d
        data_bottom = q_25 - 1.5 * d
        data_bottom = max(data_bottom,0.001)
    # print('{}:数据上下界({:.3f},{:.3f})'.format(col,data_bottom,data_top))
    # print('异常值的个数：', len(df[col][(df[col] > data_top) | (df[col] <= data_bottom)]))
    # 置为缺失值
    df[col][(df[col] > data_top) | (df[col] <= data_bottom)] = None


# 预处理4 数据缺失处理，异常处理
def data_clean(file_in, file_out, flag_save=False, fill_method = 'nearest'):
    pd.set_option('mode.chained_assignment', None)
    # 缺失值、异常值处理
    data_df = pd.read_csv(file_in, encoding='utf-8', dtype=object)
    print('数据缺失处理...数据总数:{}'.format(len(data_df)))
    # print()
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
    for fac in factors:
        data_df[fac] = data_df[fac].astype(float)

    # 几个特殊的异常处理
    data_df['总磷'] = np.where(data_df['总磷'] < 0, data_df['总磷'] * -1, data_df['总磷'])  # 负数转正
    data_df['总磷'][data_df['总磷'] == 0] = None  # 总磷=0：视为缺失值，和缺失值一起处理
    data_df['总磷'] = np.where(data_df['总磷'] > 5, data_df['总磷'] * 0.1, data_df['总磷'])  # 总磷>5：小数点左移一位

    data_df['氨氮'] = np.where(data_df['氨氮'] < 0, data_df['氨氮'] * -1, data_df['氨氮'])  # 负数转正

    data_df['高锰酸盐指数'] = np.where(data_df['高锰酸盐指数'] < 0, data_df['高锰酸盐指数'] * -1,
                                 data_df['高锰酸盐指数'])  # 负数转正

    # 四分法处理异常值（设为缺失值）
    for fac in factors:
        strange_data(data_df, fac)

    # data_df['pH值'] = np.where(data_df['pH值'] > 50, data_df['pH值'] * 0.1, data_df['pH值'])  # 50<pH值<90：小数点左移一位
    # data_df['pH值'][data_df['pH值'] > 14] = None  # pH值>14：视为缺失值，和缺失值一起处理
    # data_df['pH值'][data_df['pH值'] < 1] = None  # pH值<1：视为缺失值，和缺失值一起处理
    # strange_data(data_df, 'pH值')
    data_df['phFlag'][data_df['pH值'].isna()] = 1
    data_df['pH值'].interpolate(method=fill_method, inplace=True) # 用前一时刻的值填充缺失值

    # data_df['水温'][data_df['水温'] == 0] = None  # 水温=0：视为缺失值，和缺失值一起处理
    # strange_data(data_df, '水温')
    data_df['temperFlag'][data_df['水温'].isna()] = 1
    data_df['水温'].interpolate(method=fill_method, inplace=True)  # 用前一时刻的值填充缺失值

    # data_df['浑浊度'][data_df['浑浊度'] == 0] = None
    # strange_data(data_df, '浑浊度')
    data_df['turbiFlag'][data_df['浑浊度'].isna()] = 1
    data_df['浑浊度'].interpolate(method=fill_method, inplace=True)  # 用前一时刻的值填充缺失值

    # data_df['溶解氧'][data_df['溶解氧'] == 0] = None
    # data_df['溶解氧'][data_df['溶解氧'] >= 14.64] = None  # 溶解氧>14.64：视为缺失值，和缺失值一起处理
    data_df['doxygenFlag'][data_df['溶解氧'].isna()] = 1
    data_df['溶解氧'].interpolate(method=fill_method, inplace=True)  # 用前一时刻的值填充缺失值

    # data_df['电导率'][data_df['电导率'] == 0] = None
    data_df['conductFlag'][data_df['电导率'].isna()] = 1
    data_df['电导率'].interpolate(method=fill_method, inplace=True)  # 用前一时刻的值填充缺失值
    # data_df['电导率'][data_df['电导率'].isna()] = 439.762  # 插值无法处理的地方，设为平均值

    # data_df.reset_index(inplace=True)
    # data_df['总氮'][data_df['总氮'] <= 0] = None  # 总氮=0：视为缺失值，和缺失值一起处理
    # data_df['总氮'][data_df['总氮'] > 100] = None  # 总氮>100：视为缺失值，和缺失值一起处理
    data_df['TNFlag'][data_df['总氮'].isna()] = 1
    data_df['总氮'].interpolate(method=fill_method, inplace=True)  # 用前一时刻的值填充缺失值
    data_df['总氮'][data_df['总氮'].isna()] = 6.4186 # 插值无法处理的地方，设为平均值

    # data_df['氨氮'][data_df['氨氮'] == 0] = None  # 氨氮=0：视为缺失值，和缺失值一起处理
    # data_df['氨氮'] = np.where(data_df['氨氮'] > data_df['总氮'], None, data_df['氨氮'])
    data_df['NHFlag'][data_df['氨氮'].isna()] = 1
    data_df['氨氮'].interpolate(method=fill_method, inplace=True)  # 用前一时刻的值填充缺失值


    data_df['TPFlag'][data_df['总磷'].isna()] = 1
    data_df['总磷'].interpolate(method=fill_method, inplace=True)  # 用前一时刻的值填充缺失值

    # data_df['高锰酸盐指数'][data_df['高锰酸盐指数'] == 0] = None  # 高锰酸盐指数=0：视为缺失值，和缺失值一起处理
    data_df['permangaFlag'][data_df['高锰酸盐指数'].isna()] = 1
    data_df['高锰酸盐指数'].interpolate(method=fill_method, inplace=True)  #

    # data_df['高锰酸盐指数'][data_df['高锰酸盐指数'].isna()] = 5 #无法插值的地方，使用平均值替代。高锰酸盐的均值差不多是5
    data_df.reset_index(inplace=True)

    # 有些算法首尾无法插值得到，于是用邻近点补齐
    for fac in factors:
        data_df[fac].interpolate(method='nearest', inplace=True)

    data_df = data_df.round(3)

    if flag_save:
        data_df.to_csv(file_out, header=True, index=False, encoding='utf_8_sig')
    else:
        data_df.to_csv(file_out, header=True, index=False, encoding='utf_8_sig')
    print('数据缺失处理完成')

    # print(data_df.describe())
    statistic(data_df)


def statistic(df):
    for fac in factors:
        print("{}:\tmean:{:.4f}\tmin:{:.4f}\tmax:{:.4f}".format(fac,df[fac].mean(),df[fac].min(),df[fac].max()))


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


# 将数据转为一个个序列，用于可视化
def conver_to_seq_csv(input,output):
    xlen = 0
    csv_str = ""

    df = pd.read_csv(input, usecols=[0, 1, 2, 3, 4, 5, 6, 9, 11])

    for site in ids_shangban:
        for fac in factors_use:
            sub_df = df[fac][df['站点名称'] == site].astype(str)

            xlen = len(sub_df) if xlen < len(sub_df) else xlen
            sub_list = sub_df.tolist()
            line = ",".join(sub_list) + '\n'
            csv_str += line

    xseq = [str(i) for i in range(xlen)]
    xseq = ",".join(xseq) + '\n'
    csv_str = xseq + csv_str
    with open(output, 'w') as f:
        f.write(csv_str)



if __name__ == '__main__':
    # 预处理监测站点数据
    ######## 上坂
    # # 步骤1：
    # root = r"E:\project\mvp\Graph-WaveNet\data\water\shangban\origin"
    # data_fix_concat(root,'xls','./temp/1.csv')
    #
    # # 步骤2：
    # data_transpose('./temp/1.csv', './temp/2.csv')
    # # 步骤3：
    # data_insert_time('./temp/2.csv','./temp/3.csv','2020-01-01 00:00:00', '2020-10-31 23:00:00','4H')
    # # 步骤4
    # data_clean('./temp/3.csv', './temp/4.csv',False,'linear')
    # 可视化
    # conver_to_seq_csv('./temp/4.csv', './temp/5.csv')

    # ######### 长泰
    # 步骤1：
    root = r"E:\project\mvp\Graph-WaveNet\data\water\changtai\origin"
    data_fix_concat(root, 'csv', './temp/a.csv')

    # 步骤2：
    data_transpose('./temp/a.csv', './temp/b.csv')
    # 步骤3：
    data_insert_time('./temp/b.csv', './temp/c.csv', '2019-10-01 00:00:00', '2021-09-30 23:00:00', '4H')
    # 步骤4
    data_clean('./temp/c.csv', './temp/d.csv',False,'linear')
    # 可视化
    # conver_to_seq_csv('./temp/d.csv', './temp/e.csv')






