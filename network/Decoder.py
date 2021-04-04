import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn

from .GRUCell import GRUCell

class Decoder(nn.Module):
    def __init__(self, hidden_size:int, output_size:int, num_layers:int, device:str, proj:bool=True):
        """
        上一时刻预测结果将作为本次预测的输入，所以输入输出必须等大小
        参数：
        hidden_size      隐藏状态尺寸
        output_size      输出维数
        num_layers       GRU层数
        proj             是否进行投影，仅在隐藏尺寸和输出维数相同时可以不投影
        proj_settings    投影层设置，字典形式。若proj参数为False将被忽略
        **kwargs         GRU的其他参数
        """
        super(Decoder,self).__init__()
        self.device = device
        assert hidden_size==output_size or proj, "若不使用投影层，输出维度与隐藏状态尺寸必须保持一致。"
        self._num_layers = num_layers
        self._output_size = output_size
        self._hidden_size = hidden_size


        #除了第一层输入维数为output_size，输出维数为hidden_size
        #之后的每一层的GRU的输入输出维数相同为hidden_size
        #所有GRU上一层的输出作为下一层的输入，隐藏状态来自自身的上一时刻输出
        self.GRUs = nn.ModuleList([ GRUCell(output_size, hidden_size) if i == 0
                                   else GRUCell(hidden_size, hidden_size)
                                   for i in range(self._num_layers)])

        self.projection_layer = nn.Linear(self._hidden_size, self._output_size) if proj else None

    def forward(self, g, input, hx):
        """
        隐藏状态来自同层Encoder最后一个GRU的，最后一次隐藏状态
        参数：
        g           dgl.graph                       图结构
        input       [N, output_dim]                 输入值
        hx          [num_layers, N, hidden_size]    隐藏状态（不能为空）

        输出：
        预测值，隐藏状态
        [N, output_size], [num_layers, N, hidden_size]
        """
        hidden_states = []

        output = input
        for i, gru in enumerate(self.GRUs):
            next_hidden_state = gru(g, output, hx[i])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        #投影
        if self.projection_layer is not None:
            output = self.projection_layer(output)

        return output, torch.stack(hidden_states)