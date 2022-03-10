import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
# from dtw import dtw
import numpy as np
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import water.config as Config


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()


class nconvGLM(nn.Module):
    def __init__(self):
        super(nconvGLM,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,nvw->ncwl',(x,A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)





class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True,num_nodes=10):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        device = Config.device
        self.weightLeft = nn.Parameter(torch.randn(num_nodes, num_nodes).to(device), requires_grad=True).to(device)
        self.weight = nn.Parameter(torch.randn(in_features, out_features).to(device), requires_grad=True).to(device)
        # self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # # AXW
        # support = torch.einsum('ncvl,lo->ncvo', (input, self.weight))
        # output  = torch.einsum('wv,ncvl->ncwl', (adj[0],support))

        # # WAXW
        # adj = torch.mm(self.weightLeft, adj[0])
        # support = torch.einsum('ncvl,lo->ncvo', (input, self.weight))
        # output = torch.einsum('wv,ncvl->ncwl', (adj[0], support))

        # WAX
        adj = torch.mm(self.weightLeft, adj[0])
        output = torch.einsum('wv,ncvl->ncwl', (adj, input))

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len,order,num_nodes,device):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order
        # self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 11).to(device), requires_grad=True).to(device)
        self.weight = nn.Parameter(torch.randn(num_nodes, 11).to(device), requires_grad=True).to(device)

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gcnGLM(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len,order,num_nodes,device):
        super(gcnGLM,self).__init__()
        self.nconvGLM = nconvGLM()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order
        # self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 11).to(device), requires_grad=True).to(device)
        self.weight = nn.Parameter(torch.randn(num_nodes, 11).to(device), requires_grad=True).to(device)

    def forward(self,x,adjs):
        out = [x]
        x1 = self.nconvGLM(x,adjs)
        out.append(x1)
        for k in range(2, self.order + 1):
            x2 = self.nconvGLM(x1,adjs)
            out.append(x2)
            x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GLM(nn.Module):
    def __init__(self, in_dim,num_node,input_length):
        super(GLM, self).__init__()
        self.in_dim = in_dim
        self.num_node = num_node
        self.input_length = input_length

        self.hidden = 32

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=self.hidden,
                                    kernel_size=(1,1))
        self.convs = nn.ModuleList()

        for i in range(6):
            self.convs.append(nn.Conv2d(in_channels=self.hidden,out_channels=self.hidden,kernel_size=(1,2),dilation=(1,1)))
            self.convs.append(nn.Conv2d(in_channels=self.hidden,out_channels=self.hidden,kernel_size=(1,2),dilation=(1,3)))
        self.end_mlp = nn.Linear(self.hidden,self.num_node)

        # 构造全图
        # self.mlp1 = nn.Linear(input_length,1)
        # self.mlp2 = nn.Linear(self.hidden,64)
        # self.mlp3 = nn.Linear(64,self.num_node)

        # 只构造次对角线
        self.mlp1 = nn.Linear(input_length,1)
        self.mlp2 = nn.Linear(self.num_node,1)
        self.mlp3 = nn.Linear(self.hidden,self.num_node -1)

    def forward(self, input):
        # input = nn.functional.pad(input, (1, 0))
        # x = self.start_conv(input)
        # for i in range(12):
        #     x = self.convs[i](x)
        # x = x.squeeze(dim=3)
        # x = x.transpose(1, 2)
        # x = self.end_mlp(x)
        # x = F.relu(x)

        # mlp构造全图
        # x = F.relu(self.start_conv(input))
        # x = F.relu(self.mlp1(x))
        # x = x.squeeze(dim=3)
        # x = x.transpose(1,2)
        # x = F.relu(self.mlp2(x))
        # x = F.relu(self.mlp3(x))
        # x = F.dropout(x, 0.3, training=self.training)

        # mlp构造次对角线
        x = F.relu(self.start_conv(input))
        x = F.relu(self.mlp1(x))
        x = x.squeeze(dim=3)
        # x = x.transpose(1, 2)
        x = F.relu(self.mlp2(x))
        x = x.squeeze(dim=2)
        x = F.relu(self.mlp3(x))
        x = F.dropout(x, 0.3, training=self.training)

        # 生成矩阵
        batch_size = Config.batch_size
        device = Config.device
        x_matrix = torch.zeros([batch_size,self.num_node,self.num_node]).to(device)
        for i in range(self.num_node - 1):
            x_matrix[:,i,i+1] = x[:,i]
        adj = torch.zeros([batch_size,self.num_node,self.num_node],requires_grad=True).to(device)
        adj = adj + x_matrix

        return adj



class gcnWeight(nn.Module):
    # 用于描述因子间相互作用的邻接矩阵
    # vec_length = 16
    # nodevec1 = nn.Parameter(torch.randn(Config.num_factors, vec_length).to(Config.device), requires_grad=True).to(Config.device)
    # nodevec2 = nn.Parameter(torch.randn(vec_length, Config.num_factors).to(Config.device), requires_grad=True).to(Config.device)
    # weight_factor = nn.Parameter(torch.randn(Config.num_factors, Config.num_factors).to(Config.device),
    #                                   requires_grad=True).to(Config.device)

    def __init__(self,c_in,c_out,dropout,support_len,order,num_nodes,device,gcn_site_type=True,factor_masks=[]):
        super(gcnWeight,self).__init__()
        self.nconv = nconv()
        if (not Config.fac_single) and Config.subGraph:
            c_in = (1 + 1 + 1+4)*c_in
        else:
            c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order
        self.gcn_site_type = gcn_site_type # 等于True表示这是用于站点内的因子融合，FALSE表示这是用于不同站点的因子融合
        self.weight = nn.Parameter(torch.randn(num_nodes, num_nodes).to(device), requires_grad=True).to(device)


        # 用于描述因子间相互作用的邻接矩阵
        if (not Config.fac_single) and Config.subGraph:
            self.factor_masks = factor_masks

            vec_length = 32
            # 6个因子，每个因子都需要一个权重
            self.weight_cross = []
            for i in range(Config.num_factors):
                self.weight_cross.append(nn.Parameter(torch.randn(num_nodes, num_nodes).to(device), requires_grad=True).to(device))

            # self.nodevec1 = nn.Parameter(torch.randn(Config.num_factors, vec_length).to(device), requires_grad=True).to(device)
            # self.nodevec2 = nn.Parameter(torch.randn(vec_length, Config.num_factors).to(device), requires_grad=True).to(device)
            self.weight_insider = nn.Parameter(torch.randn(Config.num_factors, Config.num_factors).to(device), requires_grad=True).to(device)
            # self.lamb = nn.Parameter(torch.randn(1, 1).to(device), requires_grad=True).to(device)
            # self.factor_adj = np.kron(np.eye(Config.num_nodes,dtype=int),np.zeros((Config.num_factors,Config.num_factors)))

    def forward(self,x,support,dtw_matrix=None):

        # 多因子时考虑因子子图!!
        if (not Config.fac_single) and Config.subGraph:
            out = [x]
            adjs = support
            x_origin = x


            wa = torch.mm(self.weight,adjs['all'])
            x1 = self.nconv(x, wa)
            out.append(x1)
            for _ in range(3):
                x2 = self.nconv(x1, wa)
                out.append(x2)
                x1 = x2

            # 每个因子按照子图计算结果
            for i in range(Config.num_factors):
                # X_out = M * A * Cover * X
                mask_adj =  self.weight_cross[i] * adjs['cross'][i] * self.factor_masks[i]
                xo = self.nconv(x,mask_adj)
                # out.append(xo)
                x = xo
            out.append(x)

            # 同一个站点内的因子子图
            x_insider = []
            for site in range(Config.num_nodes):
                begin = site * Config.num_factors
                end = begin + Config.num_factors
                # 取出一个站点的所有因子
                x_one_site = x_origin[:,:,begin:end,:]
                # 每个站点乘以同一个子图
                mask_adj = self.weight_insider * adjs['insider']
                x_insider.append(self.nconv(x_one_site,mask_adj))
            x_insider = torch.cat(x_insider,2)
            out.append(x_insider)
            h = torch.cat(out, dim=1)

            h = self.mlp(h)
            h = F.dropout(h, self.dropout, training=self.training)
            return h


            # return x
            # fac_out = []
            # adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            # wa_f = torch.mm(self.weight_factor, adp)
            # for site in range(Config.num_nodes):
            #     begin = site * Config.num_factors
            #     end = begin + Config.num_factors
            #     # 取出一个站点的所有因子
            #     x_one_site = x[:,:,begin:end,:]
            #
            #     fac_out.append(self.nconv(x_one_site,wa_f))
            # fac_out = torch.cat(fac_out,2)
            # out.append(fac_out)
            # out_add_fac = []
            # for fea in out:
            #     out_add_fac.append(fea + self.lamb * fac_out)
            #
            # h = torch.cat(out_add_fac,dim=1)
            # h = fac_out
        else:
            out = [x]
            wa = self.weight

            for a in support:
                wa = torch.mm(wa, a)

            x1 = self.nconv(x, wa)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, wa)
                out.append(x2)
                x1 = x2
            h = torch.cat(out,dim=1)

            h = self.mlp(h)
            h = F.dropout(h, self.dropout, training=self.training)
            return h


class gwnet(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True,
                 aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,blocks=6,layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.device = device

        self.adjlearn = Config.adj_learn_type

        self.filter_convs = nn.ModuleList()
        self.lstms = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        # GLM
        self.GLM = GLM(in_dim,num_nodes,24).to(device)
        self.GLMadjs = None

        # 单维度
        if in_dim == 2:
            self.blocks = 8
        # 多维度
        else:
            self.blocks = 6
            # residual_channels = 64
            # end_channels = 64

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports
        self.adj = None

        input_data_len = Config.input_data_len
        input_feature = input_data_len
        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if supports is None:
                self.supports = []
            if aptinit is None:
                vec_length = num_nodes
                if self.adjlearn == 'embed':
                    vec_length = 16
                else:
                    vec_length = num_nodes

                # 多因子子图
                # 这个是因子掩码
                self.factor_masks = []
                if (not Config.fac_single) and Config.subGraph:
                    # 这个子图用于描述站点内不同因子的相互作用
                    self.nodevec1_insider = nn.Parameter(torch.randn(Config.num_factors, vec_length).to(device), requires_grad=True).to(device)
                    self.nodevec2_insider = nn.Parameter(torch.randn(vec_length,Config.num_factors).to(device), requires_grad=True).to(device)

                    # 这些子图用于描述不同站点间同因子的相互作用，每个因子分别有一张图
                    self.nodevec1_cross = []
                    self.nodevec2_cross = []


                    for i in range(Config.num_factors):
                        # 创建子图向量
                        self.nodevec1_cross.append(nn.Parameter(torch.randn(num_nodes, vec_length).to(device), requires_grad=True).to(device))
                        self.nodevec2_cross.append(nn.Parameter(torch.randn(num_nodes, vec_length).to(device), requires_grad=True).to(device))
                        # pass
                        # 生成因子掩码
                        sub_x = np.zeros([Config.num_factors, Config.num_factors])
                        sub_x[i][i] = 1
                        mask = np.tile(sub_x, [Config.num_site, Config.num_site])
                        self.factor_masks.append(torch.from_numpy(mask).to(device).to(torch.float32))
                # else:
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, vec_length).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(vec_length, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1

            else:
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(self.blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.lstms.append(nn.LSTM(input_size=32,hidden_size=32,num_layers=2))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    if self.adjlearn == 'GLM':

                        self.gconv.append(gcnGLM(dilation_channels,residual_channels,dropout,
                                          support_len=self.supports_len,order=3,num_nodes=num_nodes,device=device))
                    elif self.adjlearn == 'weigthed' or self.adjlearn == 'weigthedOnly' \
                            or self.adjlearn == 'merge3' or self.adjlearn == 'weigthedDTW':
                        self.gconv.append(gcnWeight(dilation_channels, residual_channels, dropout,
                                                    support_len=self.supports_len, order=3, num_nodes=num_nodes,
                                                    device=device,factor_masks=self.factor_masks))
                    elif self.adjlearn == 'secondaryGraph':
                        gcn_site_type = True
                        self.gconv.append(gcnWeight(dilation_channels, residual_channels, dropout,
                                                    support_len=self.supports_len, order=3, num_nodes=num_nodes,
                                                    device=device,gcn_site_type=gcn_site_type))
                        gcn_site_type = not gcn_site_type
                    elif self.adjlearn == 'gcnOfficial':
                        self.gconv.append(GraphConvolution( input_feature,input_feature,False))
                        input_feature = input_feature - (2-i)
                    else:
                        self.gconv.append(gcn(dilation_channels, residual_channels, dropout,
                                              support_len=self.supports_len, order=3, num_nodes=num_nodes,
                                              device=device))


        if Config.use_LSTM:
            self.lstm_last = nn.LSTM(input_size=skip_channels,hidden_size=64,num_layers=2)
            self.dnn_last = nn.Linear(64,3)
        else:
            self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                        out_channels=end_channels,
                                        kernel_size=(1, 1),
                                        bias=True)

            self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                        out_channels=out_dim,
                                        kernel_size=(1, 1),
                                        bias=True)

        self.receptive_field = receptive_field


    def compute_adj(self,node1,node2):
        return F.softmax(F.relu(torch.mm(node1, node2)), dim=1)


    def forward(self, input):
        input_numpy = input.to('cpu').numpy()
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            # 多因子子图
            if (not Config.fac_single) and Config.subGraph:
                self.adj_insider = self.compute_adj(self.nodevec1_insider,self.nodevec2_insider)
                self.adj_cross = []
                for i in range(Config.num_factors):
                    self.adj_cross.append(self.compute_adj(self.nodevec1_cross[i],self.nodevec2_cross[i]))

                adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
                self.adj_subgraph = {
                    'insider':self.adj_insider,
                    'cross':self.adj_cross,
                    'all':adp
                }

            # adp = torch.triu(adp)
            elif self.adjlearn == 'weigthed' or self.adjlearn == 'embed' or self.adjlearn == 'merge3' \
                    or self.adjlearn == 'gcnOfficial':
                adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
                # adp = torch.mm(self.nodevec1, self.nodevec2)
                # adp = F.softmax(F.relu(adp),dim=1)
                new_supports = self.supports + [adp]
                # 保留
                self.adj = adp


            # GLM算法获取邻接矩阵
            elif self.adjlearn == 'GLM':
                self.GLMadjs = self.GLM(input)

            # elif self.adjlearn == 'weigthedOnly':
            #     # nothing to done
            #     pass

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)

            # filter = filter.transpose(1,3)
            # filter = self.lstms[i](filter)
            # filter = filter.transpose(1, 3)

            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and self.supports is not None:
                # 多因子子图
                if (not Config.fac_single) and Config.subGraph:
                    x = self.gconv[i](x,self.adj_subgraph)
                elif self.adjlearn == 'GLM':
                    x = self.gconv[i](x, self.GLMadjs)
                elif self.addaptadj:
                    # gcnOfficial只跑一次
                    # if self.adjlearn != 'gcnOfficial' or i == 0:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)


        # # 使用LSTM
        if Config.use_LSTM:
            x = skip.squeeze(dim=3)
            x = x.transpose(1,2)
            x = self.lstm_last(x)[0]
            x = self.dnn_last(x)
            x = x.transpose(1,2)
            x = x.unsqueeze(3)
        # 使用CNN
        else:
            x = F.relu(skip)
            x = F.relu(self.end_conv_1(x))
            x = self.end_conv_2(x)

        return x





