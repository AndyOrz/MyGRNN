import torch
import torch.nn as nn

class MyGCNConv(torch.nn.Module):
    def __init__(self, in_channels:int, out_channels:int, **kwargs):
        super(MyGCNConv,self).__init__()
        """
        极简图卷积 Linear(A x X)
        参数：
        in_channels     每个节点输入特征维度
        out_channels    每个节点输出特征维度
        **kwargs        传递给线形层的其它参数
        """
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)


    def forward(self, adj, x):
        """
        参数：
        adj [N, N]              邻接矩阵
        x   [N, in_channels]    输入节点特征
        输出：
        [N, out_channels]
        """
        return self.linear(torch.matmul(adj,x))
        