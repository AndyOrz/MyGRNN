import torch
import torch.nn as nn
from .MyGCN import MyGCNConv
from .DAAdj import DAAdj


class DyGCGRUCell(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, num_heads:int, outputadj=False, adj_settings={}, **kwargs):
        """
        Gated recurrent unit (GRU) with Graph Convolution.
        图卷积改进的GRU(自动建立动态邻接矩阵)

        参数：
        input_size      输入尺寸
        hidden_size     输出尺寸/隐藏尺寸
        num_heads       注意力头数
        outputadj       是否输出邻接矩阵
        **adj_settings  传递给动态邻接阵的其它参数
        **kwargs        传递给图卷积层的其它参数
        GraphConv的超参数：norm='both', weight=True, bias=True, activation=None, allow_zero_in_degree=False

        输出与隐藏状态之间经过线性变换
        """
        super(DyGCGRUCell, self).__init__() #父类初始化函数
        #生成卷积层
        total_input_size = input_size + hidden_size
        self.ConvIR = MyGCNConv(total_input_size, hidden_size, **kwargs)
        self.ConvIZ = MyGCNConv(total_input_size, hidden_size, **kwargs)
        self.ConvIH = MyGCNConv(total_input_size, hidden_size, **kwargs)
        #动态邻接阵
        self.AdjBuilder = DAAdj(total_input_size, num_heads, **adj_settings)
        self.outputadj = outputadj

    def forward(self, input, hx):
        """
        GRU向前传播

        参数：
        input       [N, num_feats]      输入
        hx          [N, hidden_size]    隐藏状态(不能为空)

        输出:
        [N, hidden_size]    输出值/隐藏状态，
        [N, N]              邻接阵（仅当outputadj为True时输出）
        """

        #连接输入和隐藏状态 => [N, num_feats+hidden_size]
        inputvalue = torch.cat([input, hx],dim=1)

        #生成图结构
        adj = self.AdjBuilder(inputvalue)

        r = torch.sigmoid(self.ConvIR(adj, inputvalue)) #重置门
        z = torch.sigmoid(self.ConvIZ(adj, inputvalue)) #更新门

        h = torch.tanh(self.ConvIH(adj, torch.cat([input, r * hx], dim=1))) #新记忆

        new_state = z * hx + (1.0 - z) * h #融合新记忆和旧记忆

        if self.outputadj:
            return new_state, adj
        else:
            return new_state


    def extra_repr(self) -> str:
        return '自动生成邻接矩阵'