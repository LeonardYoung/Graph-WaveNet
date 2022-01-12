
# place = 'changtai'
place = 'shangban'
seed = 42
epoch = 2000
patience = 50
batch_size = 64
device = 'cuda:1'
# device = 'cpu'
# adj_learn_type = 'embed'            # 节点嵌入算法
# adj_learn_type = 'weigthedOnly'   # 只有权重矩阵
# adj_learn_type = 'secondaryGraph'       # 子图，先用图处理同站点间的多维因子融合，再处理不同站点间同因子融合
# adj_learn_type = 'weigthedDTW'       # DTW + 权重矩阵
# adj_learn_type = 'merge3'       # 节点嵌入 + 权重矩阵 + 对称阵
# adj_learn_type = 'GLM'            # GLM
# adj_learn_type = 'gcnOfficial'       # 节点嵌入 + 权重矩阵
# 模型选择
gcn_bool = True
adj_learn_type = 'weigthed'       # 节点嵌入 + 权重矩阵
# adj_learn_type = 'embed'            # 节点嵌入算法
subGraph = True # 子图，先用图处理同站点间的多维因子融合，再处理不同站点间同因子融合
use_LSTM = True

# 输入配置
fac_index = 0 # 单因子模式下，输入的因子下标，factors = ['pH值', '总氮', '总磷', '氨氮', '水温', '浑浊度', '溶解氧', '电导率', '高锰酸盐指数']
fac_single = True # True为单因子，FALSE为多因子，每个因子是一个站点

# 输出配置
# 输出数据的保存文件夹名
out_dir = 'GCNLSTM'

num_factors = 6     # 因子数量
num_nodes = 10      # 站点数量
input_data_len = 24 # 输入数据的特征长度
if place == 'changtai':
    num_nodes = 7
elif place == 'changban':
    num_nodes = 10

def print_all():
    print(f'adj_learn_type={adj_learn_type}\ngcn_bool={gcn_bool}\nuse_LSTM={use_LSTM}')