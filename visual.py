import open3d as o3d
import numpy as np

from torch.utils.data import DataLoader
import os
import sys
root_path=os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(os.path.join(root_path))
sys.path.append(os.path.join(root_path,'baselines'))
from baselines.dataset import ModelNet40Attack, ModelNet40
def main():
    data_root='/home/jqf/桌面/benchmark_pc_attack1-master（复件）/baselines/attack_scripts/attack/results/mn40_1024/kNN/kNN-pointnet-logits_kappa=15.0-success_0.8545-rank_-1.npz'
    test_set = ModelNet40(data_root, num_points=1024,
                          normalize=False, partition='test',
                          augmentation=False)
    raw_point=test_set.data[-20,:,:] #choose point cloud
    # 创建窗口对象
    vis = o3d.visualization.Visualizer()
    # 设置窗口标题
    vis.create_window(window_name="kitti")
    # 设置点云大小
    vis.get_render_option().point_size = 1
    # 设置颜色背景为黑色
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    # 创建点云对象
    pcd = o3d.open3d.geometry.PointCloud()
    # 将点云数据转换为Open3d可以直接使用的数据类型
    pcd.points = o3d.open3d.utility.Vector3dVector(raw_point)
    # 设置点的颜色为白色
    pcd.paint_uniform_color([1, 1, 1])
    # 将点云加入到窗口中
    vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()

