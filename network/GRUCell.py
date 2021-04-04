import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn

#  图卷积GRU单元
class GRUCell(nn.Module):
    def __init__(self, input_size:int, hidden_size:int):
        """
        参数：
        input_size  输入尺寸
        hidden_size 输出尺寸/隐藏尺寸
        若是GAT，则至少还有num_heads

        输出与隐藏状态之间经过线性变换
        """
        super(GRUCell, self).__init__() #父类初始化函数
        #生成卷积层
        self.ConvIR = dglnn.GraphConv(input_size + hidden_size, hidden_size, allow_zero_in_degree=True)
        self.ConvIZ = dglnn.GraphConv(input_size + hidden_size, hidden_size, allow_zero_in_degree=True)
        self.ConvIH = dglnn.GraphConv(input_size + hidden_size, hidden_size, allow_zero_in_degree=True)


    def forward(self, g, input, hx):
        """
        Gated recurrent unit (GRU) with Graph Convolution.
        图卷积改进的GRU

        参数：
        g           dgl.graph           图结构
        input       [N, num_feats]      输入
        hx          [N, hidden_size]    隐藏状态(不能为空)

        输出:
        [N, hidden_size]
        输出值/隐藏状态
        """

        #连接输入和隐藏状态 => [N, num_feats+hidden_size]
        inputvalue = torch.cat([input, hx],dim=1)
        # print(g.device, inputvalue.device)

        r = torch.sigmoid(self.ConvIR(g, inputvalue)) #重置门
        z = torch.sigmoid(self.ConvIZ(g, inputvalue)) #更新门

        # print(inputvalue.shape)
        # print(r.shape,hx.shape)
        h = torch.tanh(self.ConvIH(g, torch.cat([input, r * hx], dim=1))) #新记忆

        new_state = z * hx + (1.0 - z) * h #融合新记忆和旧记忆

        return new_state