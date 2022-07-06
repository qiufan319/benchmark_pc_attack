import torch
from baselines.model_utils.res_gcn_torch import pointcnn
import math
import torch.nn as nn
from baselines.model_utils.res_gcn_torch import pool,res_gcn_d,res_gcn_d_last

class get_discriminator(nn.Module):
    def __init__(self,use_bn=True, use_ibn=False):
        super(get_discriminator, self).__init__()
        self.use_bn=use_bn
        self.use_ibn=use_ibn
        self.res_gcn_d=res_gcn_d()
        self.res_gcn_d_last=res_gcn_d_last()
        self.pointcnn_net=pointcnn(64)
    def forward(self,point_cloud,use_normal=False):
        xyz = point_cloud[:, :, :3]
        # point_cnn_net=pointcnn(n_cout=64,use_bn=self.use_bn, use_ibn=self.use_ibn).cuda()
        if use_normal:
            points = point_cloud[:, :, :3]
        else:

            points = self.pointcnn_net(xyz, 8, 2)

        block_num = int(math.log2(point_cloud.size(1) /64 ) /2)

        for i in range(block_num):
            xyz, points = pool(xyz, points, 8, points.size(1)// 4)
            points = self.res_gcn_d(xyz, points, 8, 64, 4)
        points = self.res_gcn_d_last(points, 1)

        return points

if __name__ == "__main__":
    # c=torch.randn(3,4)
    # print(c.size(0))
    c=torch.rand(8,1024,3).cuda()
    net=get_discriminator()
    net(c)