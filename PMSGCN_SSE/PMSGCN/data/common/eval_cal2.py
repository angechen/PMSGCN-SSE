import torch
import numpy as np
import torch.nn as nn


class weighted_mpjpe(nn.Module):

    def __init__(self,

                 joints_num=17
                 ):
        super().__init__()


        self.weighted_xyz = nn.Parameter(torch.randn(3)).cuda()
        self.weighted_joints = nn.Parameter(torch.randn(joints_num, 1)).cuda()
        self.weighted_xyz_joints = nn.Parameter(torch.randn(joints_num, 3)).cuda()
        self.dropout = nn.Dropout(0.1)

    def forward(self,pred,gt):

    #    weighted_loss = torch.mean(torch.norm((pred - gt)*self.weighted_xyz, dim=3))
      #  weighted_loss = torch.mean(torch.norm((pred - gt)*self.weighted_xyz*self.weighted_joints, dim=3))
      #  weighted_loss = torch.mean(torch.norm((pred - gt)*torch.tensor([1,1,1.1]).cuda(),dim=3))
        weighted_loss = torch.mean(torch.norm((pred-gt)*self.dropout(self.weighted_xyz_joints), dim=3))

        return weighted_loss

