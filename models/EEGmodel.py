import torch
import torch.nn as nn
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()


class STblock(nn.Module):
    def __init__(self):
        super(STblock,self).__init__()
        self.gconv = nconv()


    def forward(self,x,a):
        # 每一帧进行图卷积
        x1 = self.gconv(x,a)





class EEGmodel(nn.Module):
    def __init__(self):
        super(EEGmodel,self).__init__()
        self.blocks = 3
