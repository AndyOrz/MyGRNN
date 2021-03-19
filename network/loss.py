import torch
import torch.nn as nn
import torch.nn.functional as F

class MyLoss(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, output, Y, mask):
        return F.smooth_l1_loss(output[mask], Y)