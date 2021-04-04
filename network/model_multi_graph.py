import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn

from .Encoder import Encoder
from .Decoder import Decoder

# MyModel seq2seq 模型
class MyModel(nn.Module):
    def __init__(self,num_feats:int, output_dim:int, hidden_size:int, num_layers:int,
                 seq_len:int, horizon:int, device:str, bidirectional:bool=False):
        """
        初始化Encoder-Decoder模型
        输入数据参数：
        num_feats    输入维数
        output_dim  输出维数

        预测需求参数：
        seq_len     输入序列长度
        horizon     输出序列长度

        超参数：
        hidden_size     隐藏状态尺寸
        num_layers      网络层数

        **kwargs        其他参数
        """
        super(MyModel,self).__init__() #父类初始化函数
        self.device = device

        self._num_feats = num_feats                  #输入参数
        self._embedded_feats = 16
        self._output_dim = output_dim                #输出参数

        self._num_layers = num_layers                #超参数-网络层数
        self._hidden_size = hidden_size              #超参数-记忆尺度

        self._bidirectional = bidirectional          #是否为双向网络

        #模块
        self.features_encoder = nn.Linear(self._num_feats, self._embedded_feats) #将输入的feature做embed
        self.encoder_model = Encoder(input_size = self._embedded_feats,
                                     hidden_size = self._hidden_size,
                                     num_layers = self._num_layers,
                                     device=device)
        if self._bidirectional:
            self.reverse_encoder_model = Encoder(input_size = self._embedded_feats,
                                        hidden_size = self._hidden_size,
                                        num_layers = self._num_layers,
                                        device=device)

        self.decoder_model = Decoder(output_size = self._output_dim,
                                     hidden_size = self._hidden_size*(2 if self._bidirectional else 1),
                                     num_layers = self._num_layers,
                                     device=device)
        #可能根据预测需求变化的参数
        self.seq_len = seq_len
        self.horizon = horizon


    def forward(self, g, x, y=None, p=0):
        """
        向前传播

        参数：
        g [seq_len] -> dgl.graph
        x [seq_len, N ,num_feats] 输入序列
        y [horizon, N ,num_feats] 输出序列的真值（可选）
        p 在decode阶段给予真值的概率（默认：0）

        输出：
        torch.tensor
        [horizon, N, output_dim] 预测时间长度 * 节点数 * 输出属性数
        """
        #N是可变的，正常输入为节点数，当使用mini-batch时为节点数*batch_size
        self._N = x.shape[1]
        assert g[0].num_nodes() == self._N, "参数 g 和 x 中的节点数量不同, {}, {}".format(g[0].num_nodes() ,self._N) #debug

        # embed
        embedded_input = self.features_encoder(x)
        #编码
        encoder_hidden_state = self.encode(g, embedded_input)
        #解码
        outputs = self.decode(g, encoder_hidden_state, truth=y, p=p) #可以考虑采用最后一个输入作为启动
        return outputs


    def encode(self, g, x):
        """
        将时间序列，逐个输入编码网络中
        参数：
        g [seq_len] -> dgl.graph
        x [seq_len, N ,num_feats] 输入序列

        输出
        torch.tensor
        [num_layers, N, hidden_size*num_directions]

        """
        encoder_hidden_state = None
        for t in range(self.seq_len):
            encoder_hidden_state = self.encoder_model(g[t], x[t], encoder_hidden_state)
        
        reverse_encoder_hidden_state = None
        if self._bidirectional:            
            for t in range(self.seq_len):
                reverse_encoder_hidden_state = self.reverse_encoder_model(g[-t], x[-t], reverse_encoder_hidden_state)

        return torch.cat((encoder_hidden_state,reverse_encoder_hidden_state),dim=2) if self._bidirectional else encoder_hidden_state


    def decode(self, g, encoder_hidden_state, startup_seq=None, truth=None, p=0):
        """
        逐个生成序列
        参数：
        g [seq_len] -> dgl.graph
        encoder_hidden_state [num_layers, N, hidden_size*num_directions] 编码结果
        startup_seq          [N, output_dim]                             启动值（默认值：全0张量）

        Curriculum Learning参数（仅在训练时使用）
        truth   [horizon, N, output_dim]    真值
        p       给予真值的概率

        输出 [horizon, N, output_dim]
        """
        #若未指定启动序列，则输入全0
        #dtype和device与encoder_hidden_state相同
        decoder_input = encoder_hidden_state.new_zeros((self._N, self._output_dim)).to(self.device) if startup_seq is None else startup_seq

        decoder_hidden_state = encoder_hidden_state

        outputs = []

        for t in range(self.horizon):
            decoder_input, decoder_hidden_state = self.decoder_model(g[t], decoder_input, decoder_hidden_state)
            outputs.append(decoder_input)
            #以一定概率给予真值
            if self.training and truth is not None:
                if np.random.uniform(0, 1) < p:
                    decoder_input = truth[t]

        return torch.stack(outputs)
