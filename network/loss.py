import torch
import torch.nn as nn
import torch.nn.functional as F

class MyLoss(nn.Module):

    def __init__(self, num_feats):
        super().__init__()
        self._num_feats = num_feats

    def forward(self, predict, truth, mask):
        '''
        参数：
        predict [out_seq_len, N, num_feats]
        truth [out_seq_len, N, num_feats]

        mask [in_seq_len, N, num_feats]
        '''

        # TODO:究竟选哪几辆车算loss
        ###### 临时！ 测试用mask算法

        # 切片索引 获取mask对应的output和Y
        predict_by_mask = predict[0,mask[9,...]==0].view(-1,self._num_feats)
        truth_by_mask = truth[0,mask[9,...]==0].view(-1,self._num_feats)
        if self._num_feats == 4:
            return F.l1_loss(predict_by_mask[:,0:2], truth_by_mask[:,0:2])+F.l1_loss(predict_by_mask[:,2:4], truth_by_mask[:,2:4])*10
        else:
            return F.l1_loss(predict_by_mask, truth_by_mask)