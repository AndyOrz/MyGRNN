import torch
import torch.nn as nn
import torch.nn.functional as F

class MyLoss(nn.Module):

    def __init__(self, num_feats):
        super().__init__()
        assert num_feats>=2
        self._num_feats = num_feats

    def forward(self, predict, truth):
        '''
        参数：
        predict [N, num_feats]
        truth [N, num_feats]
        '''

        # 位置loss采用欧几里得距离，其余loss采用L1 loss
        pos_loss = ((predict[:,0]-truth[:,0])**2+(predict[:,1]-truth[:,1])**2)**0.5
        if self._num_feats >2:                        
            other_loss = F.l1_loss(predict_by_mask[:,2:4], truth_by_mask[:,2:4])            
        else:
            other_loss = 0
        return pos_loss + other_loss