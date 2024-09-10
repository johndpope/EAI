
from __future__ import absolute_import 
from __future__ import print_function  


import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math



class GraphConvolution(nn.Module):
 
    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()  
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  
        self.att = Parameter(torch.FloatTensor(node_n, node_n))  
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))  
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        print(f"GraphConvolution input shape: {x.shape}")
        print(f"Weight shape: {self.weight.shape}")
        print(f"Att shape: {self.att.shape}")
        
        support = torch.matmul(x, self.weight)
        print(f"Support shape after matmul: {support.shape}")
        
        y = torch.matmul(self.att, support)
        print(f"Output shape after att matmul: {y.shape}")
        
        if self.bias is not None:
            return y + self.bias
        else:
            return y
        
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):

        super(GC_Block, self).__init__()  
        self.in_features = in_features
        self.out_features = in_features
        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)  
        self.do = nn.Dropout(p_dropout)  
        self.act_f = nn.Tanh()  

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)  
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48):
 
        super(GCN, self).__init__()
        self.num_stage = num_stage
        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)
        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)
        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        print(f"GCN input shape: {x.shape}")
        print(f"GC1 weight shape: {self.gc1.weight.shape}")
        print(f"GC1 att shape: {self.gc1.att.shape}")
        
        y = self.gc1(x)
        print(f"After GC1 shape: {y.shape}")
        
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        print(f"After BN1 shape: {y.shape}")
        
        y = self.act_f(y)
        y = self.do(y)
        print(f"After activation and dropout shape: {y.shape}")
        
        for i, gcb in enumerate(self.gcbs):
            y = gcb(y)
            print(f"After GC_Block {i+1} shape: {y.shape}")
        
        y = self.gc7(y)
        print(f"After GC7 shape: {y.shape}")
        
        y = y + x
        print(f"Final output shape: {y.shape}")

        return y

