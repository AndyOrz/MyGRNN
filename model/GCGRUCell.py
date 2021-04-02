import torch
import torch.nn as nn
from .MyGCN import MyGCNConv


#图卷积GRU单元
class GCGRUCell(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, **kwargs):
        """
        Gated recurrent unit (GRU) with Graph Convolution.
        图卷积改进的GRU

        参数：
        input_size  输入尺寸
        hidden_size 输出尺寸/隐藏尺寸
        **kwargs    传递给图卷积层的其它参数
        GraphConv的超参数：norm='both', weight=True, bias=True, activation=None, allow_zero_in_degree=False

        输出与隐藏状态之间经过线性变换
        """
        super(GCGRUCell, self).__init__() #父类初始化函数
        #生成卷积层
        self.ConvIR = MyGCNConv(input_size + hidden_size, hidden_size, **kwargs)
        self.ConvIZ = MyGCNConv(input_size + hidden_size, hidden_size, **kwargs)
        self.ConvIH = MyGCNConv(input_size + hidden_size, hidden_size, **kwargs)

    def forward(self, adj, input, hx):
        """
        GRU向前传播

        参数：
        adj         [N, N]              图结构
        input       [N, num_feats]      输入
        hx          [N, hidden_size]    隐藏状态(不能为空)

        输出:
        [N, hidden_size]
        输出值/隐藏状态
        """

        #连接输入和隐藏状态 => [N, num_feats+hidden_size]
        inputvalue = torch.cat([input, hx],dim=1)

        r = torch.sigmoid(self.ConvIR(adj, inputvalue)) #重置门
        z = torch.sigmoid(self.ConvIZ(adj, inputvalue)) #更新门

        h = torch.tanh(self.ConvIH(adj, torch.cat([input, r * hx], dim=1))) #新记忆

        new_state = z * hx + (1.0 - z) * h #融合新记忆和旧记忆

        return new_state