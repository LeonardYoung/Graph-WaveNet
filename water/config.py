
# place = 'changtai'
place = 'shangban'
batch_size = 64
device = 'cuda:0'
gcn_bool = True
# adj_learn_type = 'embed'            # 节点嵌入算法
# adj_learn_type = 'weigthedOnly'   # 只有权重矩阵
adj_learn_type = 'weigthed'       # 节点嵌入 + 权重矩阵
# adj_learn_type = 'secondaryGraph'       # 子图，先用图处理同站点间的多维因子融合，再处理不同站点间同因子融合
# adj_learn_type = 'weigthedDTW'       # DTW + 权重矩阵
# adj_learn_type = 'merge3'       # 节点嵌入 + 权重矩阵 + 对称阵
# adj_learn_type = 'GLM'            # GLM
# adj_learn_type = 'gcnOfficial'       # 节点嵌入 + 权重矩阵

num_factors = 6     # 因子数量
num_nodes = 10
input_data_len = 24 # 输入数据的特征长度
if place == 'changtai':
    num_nodes = 7
elif place == 'changban':
    num_nodes = 10

def print_all():
    print(f'adj_learn_type={adj_learn_type}\ngcn_bool={gcn_bool}')