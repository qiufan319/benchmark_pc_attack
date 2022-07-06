import torch
import sys
import os
import torch.nn.functional as F
root_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from model.pointnet2 import PointNetSetAbstraction,PointNetFeaturePropagation,PointNetSetAbstractionMsg
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class get_gen_model(nn.Module):
    def __init__(self,bradius=1.0,num_point=1024,up_ratio=4,use_normal=False):
        super(get_gen_model,self).__init__()
        self.up_ratio=up_ratio
        self.use_normal=use_normal
        self.sa_1 = PointNetSetAbstraction(in_channel=3, npoint=num_point, radius=bradius * 0.05,
                                      nsample=32, mlp=[32, 32, 64], group_all=False)
        self.sa_2 = PointNetSetAbstraction(in_channel=64 + 3, npoint=int(num_point / 2), radius=bradius * 0.01,
                                      nsample=32, mlp=[64, 64, 128], group_all=False)
        self.sa_3 = PointNetSetAbstraction(in_channel=128 + 3, npoint=int(num_point / 4), radius=bradius * 0.2,
                                      nsample=32, mlp=[128, 128, 256], group_all=False)
        self.sa_4 = PointNetSetAbstraction(in_channel=256 + 3, npoint=int(num_point / 8), radius=bradius * 0.3,
                                      nsample=32, mlp=[256, 256, 512], group_all=False)
        self.pfp_1 = PointNetFeaturePropagation(in_channel=512, mlp=[64])
        self.pfp_2 = PointNetFeaturePropagation(in_channel=256, mlp=[64])
        self.pfp_3 = PointNetFeaturePropagation(in_channel=128, mlp=[64])
        self.conv1d_2 = nn.Conv1d(379, 256, kernel_size=1)
        self.conv1d_3= nn.Conv1d(296, 128, kernel_size=1)
        self.conv1d_4 = nn.Conv1d(168, 64, kernel_size=1)
        self.conv1d_5 = nn.Conv1d(64, 3, kernel_size=1)
        self.relu=nn.ReLU()
    def forward(self, point_cloud,labels_onehot):
        batch_size = point_cloud.size(0)
        num_point = point_cloud.size(1)
        l0_xyz = point_cloud[:, :, :3]
        if self.use_normal:
            l0_points = point_cloud[:, :, :3]
        else:
            l0_points = None
        l0_xyz=l0_xyz.permute(0,2,1)

        # Layer 1
        l1_xyz, l1_points = self.sa_1(l0_xyz, l0_points)

        l2_xyz, l2_points = self.sa_2(l1_xyz, l1_points)

        l3_xyz, l3_points = self.sa_3(l2_xyz, l2_points)

        l4_xyz, l4_points = self.sa_4(l3_xyz, l3_points)
        # Feature Propagation layers

        up_l4_points = self.pfp_1(l0_xyz, l4_xyz, None, l4_points)
        up_l3_points = self.pfp_2(l0_xyz, l3_xyz, None, l3_points)
        up_l2_points = self.pfp_3(l0_xyz, l2_xyz, None, l2_points)

        labels_onehot1 = labels_onehot.unsqueeze(2).repeat([1,1, num_point])

        # up_l2_points=up_l2_points.permute(0,2,1)
        up_l2_points = torch.cat([up_l2_points, labels_onehot1],1)

        labels_onehot1 = labels_onehot.unsqueeze(2).repeat([1,1, num_point])
        # up_l3_points = up_l3_points.permute(0, 2, 1)
        up_l3_points = torch.cat([up_l3_points, labels_onehot1], 1)

        labels_onehot1 = labels_onehot.unsqueeze(2).repeat([1,1, num_point])
        # up_l4_points = up_l4_points.permute(0, 2, 1)
        up_l4_points = torch.cat([up_l4_points, labels_onehot1], 1)

        # concat features
        new_points_list = []
        for i in range(self.up_ratio):
            # if l0_xyz.size(-1) != 3:
            #     l0_xyz = l0_xyz.permute(0, 2, 1)
            # l1_points=l1_points.permute(0,2,1)
            concat_feat = torch.cat([up_l4_points, up_l3_points, up_l2_points, l1_points, l0_xyz], axis=1)

            # concat_feat = concat_feat.unsqueeze(-2)

            # concat_feat=concat_feat.permute(0,3,2,1)
            # F.relu(conv(new_points))
            concat_feat=F.relu(self.conv1d_2(concat_feat))
            # concat_feat=self.relu(concat_feat)

            labels_onehot1 = labels_onehot.unsqueeze(2).repeat(1, 1, num_point)
            concat_feat = torch.cat([concat_feat, labels_onehot1], 1)
            new_points = F.relu(self.conv1d_3(concat_feat))
            # new_points = self.conv2d_3(concat_feat)
            # new_points = self.relu(new_points)
            new_points_list.append(new_points)
        net = torch.cat(new_points_list, -1)
        labels_onehot1 = labels_onehot.unsqueeze(2).repeat(1, 1, num_point)
        net = torch.cat([net, labels_onehot1], 1)
        # get the xyz
        # coord = self.conv2d_4(net)
        # coord = self.relu(coord)
        coord=F.relu(self.conv1d_4(net))
        # coord = self.conv2d_5(coord) # B*(2N)*1*3
        # coord = self.relu(coord)
        coord = F.relu(self.conv1d_5(coord))
        # coord = coord.squeeze(2)  # B*(2N)*3
        return coord, None


def conv2d(inputs,
           num_output_channels,
           kernel_size,
           stride=[1, 1],
           padding='same',
           data_format='NHWC',
           use_xavier=True,
           activation_fn=nn.ReLU):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    data_format: 'NHWC' or 'NCHW'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  # kernel_h, kernel_w = kernel_size
  assert(data_format=='NHWC' or data_format=='NCHW')
  if data_format == 'NHWC':
    inputs=inputs.permute(0,3,2,1)
  elif data_format=='NCHW':
    inputs=inputs
  conv2d_2=nn.Conv2d(inputs.size(1),num_output_channels,kernel_size=1,padding='same',stride=1).cuda()
  inputs=inputs.permute(0,3,2,1)
  outputs =conv2d_2(inputs)

  if activation_fn is not None:
    relu = nn.ReLU()
    outputs=relu(outputs)
  return outputs
if __name__ == "__main__":
    # c=torch.randn(3,4)
    # print(c.size(0))
    pointcloud=torch.rand(4,1024,3).cuda()
    label=torch.rand(4,40).cuda()
    net = get_gen_model().cuda()
    point,_=net(pointcloud,label)

