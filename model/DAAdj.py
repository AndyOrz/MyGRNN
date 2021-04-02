import torch
import torch.nn as nn
import torch.nn.functional as F


class DAAdj(nn.Module):
	def __init__(self, num_feats, num_heads:int, negative_slope:float=0):
		"""
		Dynamic Attention Adjacency matrix
		动态注意力邻接阵
		根据节点特征建立带权重的邻接矩阵

		参数：
		num_feats	输入维数
		num_heads	注意力头数
		negative_slope LeakyReLU的负半轴斜率
		"""
		super(DAAdj,self).__init__()
		self.num_heads = num_heads
		self.num_feats = num_feats
		self.lin_dist = nn.Linear(num_feats * 2, num_heads)
		self.lin_merge = nn.Linear(num_heads, 1)
		if negative_slope == 0:
			self.leaky_relu = nn.ReLU()
		else:
			self.leaky_relu = nn.LeakyReLU(negative_slope)
	
		self.selfbias = nn.Parameter(torch.ones(num_heads,1))

		
	def forward(self, x): 
		"""
		向前传播

		参数：
		x [num_nodes, num_feats] 输入
		
		输出 邻接矩阵 [num_nodes, num_nodes]
		"""
		num_nodes, num_feats= x.shape

        # [num_nodes, num_nodes, num_feats]
		x_s = x.unsqueeze(1).repeat(1,num_nodes,1) #repeat alone axis 1
		x_t = x.unsqueeze(0).repeat(num_nodes,1,1) #repeat alone axis 0

		#某种意义上是表示vehicle 𝑗 与 vehicle 𝑖的嵌入差异（i->j）
		dist = self.lin_dist(torch.cat((x_s,x_t),2))
		dist += torch.diag_embed(self.selfbias.repeat(1, num_nodes), dim1=0, dim2=1)
		#多头注意力 [num_nodes, num_nodes, num_heads]
		heads = F.softmax(self.leaky_relu(dist),1)
		#融合注意力 [num_nodes, num_nodes, 1]
		return self.lin_merge(heads)[...,0]