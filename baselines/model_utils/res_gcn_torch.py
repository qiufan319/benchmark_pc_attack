import torch
import numpy as np
import torch.nn as nn
import torch
# import group_points_cuda
from model.pointnet2 import farthest_point_sample,index_points
def knn(k, xyz1, xyz2):
    '''
    Input:
        k: int32, number of k in k-nn search
        xyz1: (batch_size, ndataset, c) float32 array, input points
        xyz2: (batch_size, npoint, c) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''
    b = xyz1.size(0)
    n = xyz1.size(1)
    c = xyz1.size(2)
    m = xyz2.size(1)
    xyz1=xyz1.reshape(b,1,n,c).repeat(1,m,1,1)
    xyz2=xyz2.reshape(b,m,1,c).repeat(1,1,n,1)
    # xyz1 = tf.tile(tf.reshape(xyz1, (b,1,n,c)), [1,m,1,1])
    # xyz2 = tf.tile(tf.reshape(xyz2, (b,m,1,c)), [1,1,n,1])
    dist = torch.sum((xyz1-xyz2)**2, -1)
    out, outi = torch.topk(dist, k,dim=-1)
    idx = outi
    val = out
    #val, idx = tf.nn.top_k(-dist, k=k) # ONLY SUPPORT CPU
    return val, idx
def group(xyz, points, k, dilation=1, use_xyz=False):
    _,idx = knn(k*dilation+1,xyz,xyz)
    idx = idx[:, :,1::dilation]
    xyz=xyz.detach()
    grouped_xyz = index_points(xyz, idx)
    # grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx) # (batch_size, npoint, k, channel)
        if use_xyz:
            grouped_points = torch.cat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, k, 3+channel)
    else:
        grouped_points = grouped_xyz

    return grouped_xyz, grouped_points, idx

class batch_norm(nn.Module):
    def __init__(self,use_bn=True, use_ibn=False):
        super(batch_norm, self).__init__()
        self.BN=nn.BatchNorm2d(3)
        self.IBN=nn.InstanceNorm2d(3)
        self.use_bn=use_bn
        self.use_ibn=use_ibn
    def forward(self,x):
        if self.use_bn:
            return self.BN(x)
        if self.use_ibn:
            return self.IBN(x)
        return x
class pointcnn(nn.Module):
    def __init__(self,n_cout,use_bn=False, use_ibn=False):
        super(pointcnn, self).__init__()
        # self.conv1=nn.Conv2d(3,n_cout,kernel_size=1)
        self.use_bn=use_bn
        self.use_ibn=use_ibn
        self.relu=nn.ReLU()
        self.batch_norm=batch_norm(use_bn=self.use_bn, use_ibn=self.use_ibn)
        self.n_cout=n_cout
    def forward(self,xyz,k,n_blocks):
        _, grouped_points, _ = group(xyz, None, k)
        for idx in range(n_blocks):
            grouped_points = Conv2D(grouped_points,self.n_cout)
            if idx == n_blocks - 1:
                return torch.max(grouped_points, axis=2)[0]
            else:
                grouped_points = self.batch_norm(grouped_points)
                grouped_points = self.relu(grouped_points)

def Conv2D(x,out_channel):
    x=x.permute(0,3,2,1)
    in_channel=x.size(1)
    conv=nn.Conv2d(in_channel,out_channel,kernel_size=1).cuda()
    return conv(x).permute(0,3,2,1)

def pool(xyz, points, k, npoint):
    new_xyz = index_points(xyz, farthest_point_sample(xyz,npoint))
    # if new_xyz.size(-1)==3:
    #     new_xyz=new_xyz.permute(0,2,1)
    _, idx = knn(k, xyz, new_xyz)
    new_points = torch.max(index_points(points, idx), axis=2)[0]
    return new_xyz, new_points
class res_gcn_d(nn.Module):
    def __init__(self,use_bn=False, use_ibn=False, indices=None):
        super(res_gcn_d, self).__init__()
        self.batch_norm=batch_norm(use_bn=use_bn, use_ibn=use_ibn)
        self.leak_relu=nn.LeakyReLU()
    def forward(self,xyz,points,k,n_cout,n_blocks,indices=None):
        for idx in range(n_blocks):
            shortcut = points
            # Center Features
            points = self.batch_norm(points)
            points = self.leak_relu(points)

            #Neighbor Feature
            if idx == 0 and indices is None:
                _, grouped_points, indices = group(xyz, points, k)
            else:
                grouped_points = index_points(points, indices)
            # Center Conv
            center_points = points.unsqueeze(2)  # tf.expand_dims(points, axis=2)
            points = Conv2D(center_points, n_cout)
            grouped_points_nn = Conv2D(grouped_points, n_cout)
            points = torch.mean(torch.cat([points, grouped_points_nn], axis=2), axis=2) + shortcut
        return points
# def res_gcn_d(xyz, points, k, n_cout, n_blocks, use_bn=False, use_ibn=False, indices=None):
#     for idx in range(n_blocks):
#             shortcut = points
#
#             # Center Features
#             points = self.batch_norm(points)
#             leak_relu=nn.LeakyReLU().cuda()
#             points = leak_relu(points)
#             # Neighbor Features
#             if idx == 0 and indices is None:
#                 _, grouped_points, indices = group(xyz, points, k)
#             else:
#                 grouped_points = index_points(points, indices)
#             # Center Conv
#             center_points = points.unsqueeze(2) #tf.expand_dims(points, axis=2)
#             points=Conv2D(center_points,n_cout)
#             # conv1=nn.Conv2d(center_points.size(1),n_cout,kernel_size=1).cuda()
#             # points = conv1(center_points)
#             # Neighbor Conv
#             grouped_points_nn=Conv2D(grouped_points,n_cout)
#             # conv2=nn.Conv2d(grouped_points.size(1),n_cout,kernel_size=1).cuda()
#             # grouped_points_nn = conv2(grouped_points)
#             # CNN
#             points = torch.mean(torch.cat([points, grouped_points_nn], axis=2), axis=2) + shortcut
#
#     return points
class res_gcn_d_last(nn.Module):
    def __init__(self):
        super(res_gcn_d_last, self).__init__()
        self.BN=batch_norm1d().cuda()
        self.leak_relu = nn.LeakyReLU()
    def forward(self,points,n_cout):
        points = self.BN(points)
        points = self.leak_relu(points)
        center_points = points.unsqueeze(2)
        points = Conv2D(center_points, n_cout).squeeze(2)
        return points

class batch_norm1d(nn.Module):
    def __init__(self,use_bn=True, use_ibn=False):
        super(batch_norm1d, self).__init__()
        self.BN=nn.BatchNorm1d(64)
        self.IBN=nn.InstanceNorm1d(64)
        self.use_bn=use_bn
        self.use_ibn=use_ibn
    def forward(self,x):
        if self.use_bn:
            return self.BN(x)
        if self.use_ibn:
            return self.IBN(x)
        return x
# def res_gcn_d_last(points, n_cout, use_bn=False, use_ibn=False):
#         BN=batch_norm(use_bn=use_bn, use_ibn=use_ibn)
#         points = BN(points)
#         leak_relu=nn.LeakyReLU()
#         points = leak_relu(points)
#         center_points = points.unsqueeze(2)
#         points = Conv2D(center_points,n_cout).squeeze(2)#tf.squeeze(conv2d(center_points, n_cout, name='conv_center'), axis=2)
#
#         return points
if __name__ == "__main__":
    # c=torch.randn(3,4)
    # print(c.size(0))
    c=torch.rand(4,1024,3).cuda()
    pointcnn(c,None,8)
    net = pointcnn(64).cuda()
    net(c,8,2)