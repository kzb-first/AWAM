# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .WTConv import WTConv2d
from einops import rearrange
from .ASCA import ASCA

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

  

class SGAB(nn.Module):
    def __init__(self, n_feats):   
        super().__init__()
        i_feats =n_feats*2
        
        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0) 
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7//2, groups= n_feats)     
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        
        # self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        
    def forward(self, x):      
        shortcut = x.clone()
        
        #Ghost Expand      
        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1) 
        x = x*self.DWConv1(a)
        x = self.Conv2(x)
        
        return  x*self.scale + shortcut       


class WGC(nn.Module):
    def __init__(
        self, n_feats):   
        super().__init__()
        
        self.WT = WTConv2d(in_channels=n_feats,out_channels=n_feats)
        self.LFE = SGAB(n_feats)
        
    def forward(self, x): 

        
        x = self.WT(x)
        x = self.LFE(x)  
        
        return x   
    
    
class ResGroup(nn.Module):
    def __init__(self, n_resblocks, n_feats):
        super(ResGroup, self).__init__()
        self.body = nn.ModuleList([
            WGC(n_feats) \
            for _ in range(n_resblocks)])
        self.body_t = ASCA(dim=n_feats,num_heads=4)
    def forward(self, x):
        res = x.clone()
        
        for i, block in enumerate(self.body):
            res = block(res)
            
        x = self.body_t(res) + x        
        
        return x 
    


class AWAM(nn.Module):
    def __init__(self, n_resblocks=6, n_resgroups=1, n_colors=10, n_feats=60, scale=4):
        super(AWAM, self).__init__()
        
        #res_scale = res_scale
        self.n_resgroups = n_resgroups
        
        # self.sub_mean = MeanShift(1.0)   
        self.head = nn.Conv2d(n_colors, n_feats, 3, 1, 1)
        
        # define body module
        self.body = nn.ModuleList([
            ResGroup(
                n_resblocks, n_feats)
            for i in range(n_resgroups)])
        
        if self.n_resgroups > 1:
            self.body_t = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

        # define tail module
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_colors*(scale**2), 3, 1, 1),
            nn.PixelShuffle(scale)
        )
        
    def forward(self, x):
        x = self.head(x)
        res = x
        for i in self.body:
            res = i(res)
        if self.n_resgroups>1:
            res = self.body_t(res) + x
        x = self.tail(res)
        return x

def create_net():
    net = AWAM()
    return net

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择 GPU 或 CPU
    input = torch.randn(1, 10, 180, 360).to(device)  # 将输入数据转移到 GPU

    model = AWAM().to(device)  # 将模型转移到 GPU
    y = model(input)  # 在 GPU 上进行推理
    print(y.shape)

    total_params = sum(p.numel() for p in model.parameters())
    total_params_in_million = total_params / 1e3  
    print(f"Total parameters: {total_params_in_million:.2f}K")
