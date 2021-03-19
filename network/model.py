# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
import torch
import torch.nn as nn
import dgl
from dgl.nn import GATConv

# %% [markdown]
# ### GRUCell

# %%
#图卷积GRU单元
class GRUCell(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, **kwargs):
        """
        参数：
        input_size  输入尺寸
        hidden_size 输出尺寸/隐藏尺寸
        **kwargs    传递给图卷积层的其它参数
        若是GAT，则至少还有num_heads

        输出与隐藏状态之间经过线性变换
        """
        super(GRUCell, self).__init__() #父类初始化函数
        #生成卷积层
        self.ConvIR = GATConv(input_size + hidden_size, hidden_size, allow_zero_in_degree=True, **kwargs)
        self.ConvIZ = GATConv(input_size + hidden_size, hidden_size, allow_zero_in_degree=True, **kwargs)
        self.ConvIH = GATConv(input_size + hidden_size, hidden_size, allow_zero_in_degree=True, **kwargs)


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

        r = torch.sigmoid(self.ConvIR(g, inputvalue)) #重置门
        z = torch.sigmoid(self.ConvIZ(g, inputvalue)) #更新门

        print(inputvalue.shape)
        print(r.shape,hx.shape)
        h = torch.tanh(self.ConvIH(g, torch.cat([input, r * hx], dim=1))) #新记忆

        new_state = z * hx + (1.0 - z) * h #融合新记忆和旧记忆

        return new_state

# %% [markdown]
# ## Encoder

# %%
class Encoder(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, num_layers:int, **kwargs):
        """
        使用GRUCell的编码器

        参数：
        input_size       输入维数
        hidden_size      隐藏状态尺寸
        num_layers       GRU层数
        **kwargs         GRU的其他参数
        """
        super(Encoder, self).__init__()
        self._num_layers = num_layers
        self._hidden_size = hidden_size

        #除了第一层输入维数为input_size，输出维数为hidden_size
        #之后的每一层的GRU的输入输出维数相同为hidden_size
        #所有GRU上一层的输出作为下一层的输入，隐藏状态来自自身的上一时刻输出
        self.GRUs = nn.ModuleList([ GRUCell(input_size, hidden_size, **kwargs) if i == 0
                                   else GRUCell(hidden_size, hidden_size, **kwargs) 
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
            hx = input.new_zeros((self._num_layers, input.shape[0], self._hidden_size))

        hidden_states = []
        output = input

        #以上一层的输出为下一层输入
        #每个GRU的隐藏状态来自自身的上一时刻输出
        for i, gru in enumerate(self.GRUs):
            next_hidden_state = gru(g, output, hx[i])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return torch.stack(hidden_states)

# %% [markdown]
# ## Decoder

# %%
class Decoder(nn.Module):
    def __init__(self, hidden_size:int, output_size:int, num_layers:int, proj:bool=True, proj_settings={'bias':True} ,**kwargs):
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
        assert hidden_size==output_size or proj, "若不使用投影层，输出维度与隐藏状态尺寸必须保持一致。"
        self._num_layers = num_layers
        self._output_size = output_size
        self._hidden_size = hidden_size


        #除了第一层输入维数为output_size，输出维数为hidden_size
        #之后的每一层的GRU的输入输出维数相同为hidden_size
        #所有GRU上一层的输出作为下一层的输入，隐藏状态来自自身的上一时刻输出
        self.GRUs = nn.ModuleList([ GRUCell(output_size, hidden_size, **kwargs) if i == 0
                                   else GRUCell(hidden_size, hidden_size, **kwargs)
                                   for i in range(self._num_layers)])
        
        self.projection_layer = nn.Linear(self._hidden_size, self._output_size, **proj_settings) if proj else None

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

# %% [markdown]
# # MyModel

# %%
#seq2seq 模型
class MyModel(nn.Module):
    def __init__(self,num_feats:int, output_dim:int, hidden_size:int, num_layers:int,
                 seq_len:int, horizon:int, **kwargs):
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

        self._num_feats = num_feats                  #输入参数
        self._output_dim = output_dim                #输出参数

        self._num_layers = num_layers                #超参数-网络层数
        self._hidden_size = hidden_size              #超参数-记忆尺度
        
        #模块
        self.encoder_model = Encoder(input_size = self._num_feats,
                                     hidden_size = self._hidden_size,
                                     num_layers = self._num_layers, **kwargs)

        self.decoder_model = Decoder(output_size = self._output_dim,
                                     hidden_size = self._hidden_size,
                                     num_layers = self._num_layers, **kwargs)
        #可能根据预测需求变化的参数
        self.seq_len = seq_len 
        self.horizon = horizon 


    def forward(self, g, x, y=None, p=0):
        """
        向前传播

        参数：
        g dgl.graph
        x [seq_len, N ,num_feats] 输入序列
        y [horizon, N ,num_feats] 输出序列的真值（可选）
        p 在decode阶段给予真值的概率（默认：0）

        输出：
        torch.tensor
        [horizon, N, output_dim] 预测时间长度 * 节点数 * 输出属性数
        """
        #N是可变的，正常输入为节点数，当使用mini-batch时为节点数*batch_size
        self._N = x.shape[1]
        assert g.num_nodes() == self._N, "参数 g 和 x 中的节点数量不同, {}, {}".format(g.num_nodes() ,self._N) #debug

        #编码
        encoder_hidden_state = self.encode(g, x)
        #解码
        outputs = self.decode(g, encoder_hidden_state, truth=y, p=p) #可以考虑采用最后一个输入作为启动
        return outputs


    def encode(self, g, x):
        """
        将时间序列，逐个输入编码网络中
        参数：
        g dgl.graph
        x [seq_len, N ,num_feats] 输入序列

        输出
        torch.tensor
        [num_layers, N, hidden_size]

        """
        encoder_hidden_state = None
        for t in range(self.seq_len):
            encoder_hidden_state = self.encoder_model(g, x[t], encoder_hidden_state)
            
        return encoder_hidden_state


    def decode(self, g, encoder_hidden_state, startup_seq=None, truth=None, p=0):
        """
        逐个生成序列
        参数：
        g dgl.graph
        encoder_hidden_state [num_layers, N, hidden_size] 编码结果
        startup_seq          [N, output_dim]              启动值（默认值：全0张量）

        Curriculum Learning参数（仅在训练时使用）
        truth   [horizon, N, output_dim]    真值
        p       给予真值的概率

        输出 [horizon, N, output_dim]
        """
        #若未指定启动序列，则输入全0
        #dtype和device与encoder_hidden_state相同
        decoder_input = encoder_hidden_state.new_zeros((self._N, self._output_dim)) if startup_seq is None else startup_seq

        decoder_hidden_state = encoder_hidden_state

        outputs = []

        for t in range(self.horizon):
            decoder_input, decoder_hidden_state = self.decoder_model(g, decoder_input, decoder_hidden_state)
            outputs.append(decoder_input)
            #以一定概率给予真值
            if self.training and truth is not None:
                if np.random.uniform(0, 1) < p:
                    decoder_input = truth[t]

        return torch.stack(outputs)


