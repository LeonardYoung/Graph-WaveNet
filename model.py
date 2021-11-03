import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


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


class gcnWeight(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len,order,num_nodes,device):
        super(gcnWeight,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order
        self.weight = nn.Parameter(torch.randn(num_nodes, num_nodes).to(device), requires_grad=True).to(device)

    def forward(self,x,support):
        out = [x]
        for a in support:
            wa = torch.mm(self.weight,a)
            # wa = a * self.weight
            x1 = self.nconv(x,wa)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,wa)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


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
        self.mlp1 = nn.Linear(input_length,1)
        self.mlp2 = nn.Linear(self.hidden,64)
        self.mlp3 = nn.Linear(64,self.num_node)

        self.mlp4 = nn.Linear(self.in_dim * self.input_length,256)
        self.mlp5 = nn.Linear(256,64)
        # self.mlp6 = nn.Linear(128,64)
        # self.mlp7 = nn.Linear(64,32)
        self.mlp8 = nn.Linear(64,self.num_node)

    def forward(self, input):

        # x = F.relu(self.start_conv(input))
        # x = F.relu(self.mlp1(x))
        # x = x.squeeze(dim=3)
        # x = x.transpose(1,2)
        # x = F.relu(self.mlp2(x))
        # x = F.relu(self.mlp3(x))


        x = input.reshape(input.shape[0],self.num_node,-1)
        x = F.relu(self.mlp4(x))
        x = F.relu(self.mlp5(x))
        # x = F.relu(self.mlp6(x))
        # x = F.relu(self.mlp7(x))
        x = F.relu(self.mlp8(x))


        x = F.dropout(x, 0.3, training=self.training)

        return x


class gwnet(nn.Module):
    def __init__(self, adjlearn,device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True,
                 aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,blocks=6,layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.device = device
        self.adjlearn = adjlearn

        self.filter_convs = nn.ModuleList()
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
                    elif self.adjlearn == 'weigthed' or self.adjlearn == 'weigthedOnly':
                        self.gconv.append(gcnWeight(dilation_channels, residual_channels, dropout,
                                                    support_len=self.supports_len, order=3, num_nodes=num_nodes,
                                                    device=device))
                    else:
                        self.gconv.append(gcn(dilation_channels, residual_channels, dropout,
                                              support_len=self.supports_len, order=3, num_nodes=num_nodes,
                                              device=device))


        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field

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
            # 对称阵算法
            # m1 = torch.tanh(0.25 * self.nodevec1)
            # m2 = torch.tanh(0.25 * self.nodevec2)
            # adp = F.relu(torch.tanh(torch.mm(m1, m2.t()) - torch.mm(m2, m1.t())))
            # adp = F.softmax(F.relu(torch.tanh(torch.mm(m1, m2.t()) - torch.mm(m2, m1.t()))), dim=1)
            # adp = torch.triu(adp)
            if self.adjlearn == 'weigthed' or self.adjlearn == 'embed':
                adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
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
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
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
                if self.adjlearn == 'GLM':
                    x = self.gconv[i](x, self.GLMadjs)
                elif self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x





