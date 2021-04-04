import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn

from .GRUCell import GRUCell


# Encoder
class Encoder(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, num_layers:int, device:str):
        """
        使用GRUCell的编码器

        参数：
        input_size       输入维数
        hidden_size      隐藏状态尺寸
        num_layers       GRU层数
        **kwargs         GRU的其他参数
        """
        super(Encoder, self).__init__()
        self.device = device
        self._num_layers = num_layers
        self._hidden_size = hidden_size

        #除了第一层输入维数为input_size，输出维数为hidden_size
        #之后的每一层的GRU的输入输出维数相同为hidden_size
        #所有GRU上一层的输出作为下一层的输入，隐藏状态来自自身的上一时刻输出
        self.GRUs = nn.ModuleList([ GRUCell(input_size, hidden_size) if i == 0
                                   else GRUCell(hidden_size, hidden_size)
                                   for i in range(self._num_layers)])


    def forward(self, g, input, hx=None):
        """
        每一层GRU以上一层GRU的输出作为输入，
        每一时刻输入获得的GRU的隐藏状态传递给同一GRU的下一时刻
        参数：
        g           dgl.graph                       图结构
        input       [N, num_feats]                  输入值
        hx          [num_layers, N, hidden_size]    隐藏状态（多出一维是因为每一层GRU都有一个隐藏状态）

        输出
        [num_layers, N, hidden_size]
        """
        #对序列首个输入进行初始化
        if hx is None:
            hx = torch.zeros((self._num_layers, input.shape[0], self._hidden_size)).to(self.device)

        hidden_states = []
        output = input

        #以上一层的输出为下一层输入
        #每个GRU的隐藏状态来自自身的上一时刻输出
        for i, gru in enumerate(self.GRUs):
            next_hidden_state = gru(g, output, hx[i])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return torch.stack(hidden_states)