import numpy as np
import pyshtools as pysh
import torch
import matplotlib as plt
# import plotly.graph_objects as go
from cv2 import getGaussianKernel


# from plotly.subplots import make_subplots

# def plot_pc(pc, second_pc=None, s=4, o=0.6):
#     fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],)
#     fig.add_trace(
#         plt.Scatter3d(x=pc[:, 0], y=pc[:, 1], z=pc[:, 2], mode='markers', marker=dict(size=s, opacity=o)),
#         row=1, col=1
#     )
#     if second_pc is not None:
#         fig.add_trace(
#             go.Scatter3d(x=second_pc[:, 0], y=second_pc[:, 1], z=second_pc[:, 2], mode='markers', marker=dict(size=s, opacity=o)),
#             row=1, col=2
#         )
#     fig.update_layout(scene_aspectmode='data')
#     fig.show()

def convert_pc_to_grid(pc, lmax, device="cuda"):
    """ Function for projecting a point cloud onto the surface of the unit sphere centered at its centroid. """

    pc =pc.to(device)

    grid = pysh.SHGrid.from_zeros(lmax, grid='DH')
    nlon = grid.nlon
    nlat = grid.nlat
    ngrid = nlon * nlat
    grid_lon = torch.from_numpy(np.linspace(0, nlon * np.pi * 2 / (nlon - 1), num=nlon, endpoint=False)).to(device)
    grid_lat = torch.from_numpy(np.linspace(nlat // 2 * np.pi / -(nlat - 1), np.pi / 2, num=nlat, endpoint=True)).to(
        device)
    grid_lon = torch.broadcast_to(grid_lon, (nlat, nlon))
    grid_lat = torch.broadcast_to(grid_lat.view(nlat, 1), (nlat, nlon))
    grid_lon = grid_lon.reshape(1, ngrid)
    grid_lat = grid_lat.reshape(1, ngrid)

    origin = torch.mean(pc, axis=0)  # the center of the unit sphere
    pc -= origin  # for looking from the perspective of the origin
    npc = len(pc)
    origin = origin.to("cpu").numpy()

    pc_x, pc_y, pc_z = pc[:, 0], pc[:, 1], pc[:, 2]

    pc_r = torch.sqrt(pc_x * pc_x + pc_y * pc_y + pc_z * pc_z)
    pc_lat = torch.arcsin(pc_z / pc_r)
    pc_lon = torch.atan2(pc_y, pc_x)
    pc_r = pc_r.view(npc, 1)
    pc_lat = pc_lat.view(npc, 1)
    pc_lon = pc_lon.view(npc, 1)

    dist = -torch.cos(grid_lat) * torch.cos(pc_lat) * torch.cos(grid_lon - pc_lon) + torch.sin(grid_lat) * torch.sin(
        pc_lat)

    argmin = torch.argmin(dist, axis=0)
    grid_r = pc_r[argmin].view(nlat, nlon)
    grid.data = grid_r.to("cpu").numpy()  # data of the projection onto the unit sphere

    argmin = torch.argmin(dist, axis=1)  # argmin on a different axis
    flag = torch.zeros(ngrid, dtype=bool)
    flag[argmin] = True  # indicates the polar angles for which the grid data can be interpreted as a point
    flag = flag.to("cpu").numpy()

    return grid, flag, origin


def convert_pc_to_grid_np(pc, lmax):
    """ Function for projecting a point cloud onto the surface of the unit sphere centered at its centroid. """

    pc = np.copy(pc)  # for not changing the original input point cloud

    grid = pysh.SHGrid.from_zeros(lmax, grid='DH')
    nlon = grid.nlon
    nlat = grid.nlat
    ngrid = nlon * nlat

    grid_lon = np.linspace(0, nlon * np.pi * 2 / (nlon - 1), num=nlon, endpoint=False)
    grid_lat = np.linspace(nlat // 2 * np.pi / -(nlat - 1), np.pi / 2, num=nlat, endpoint=True)

    grid_lon = np.broadcast_to(grid_lon, (nlat, nlon))
    grid_lat = np.broadcast_to(grid_lat.reshape((nlat, 1)), (nlat, nlon))

    grid_lon = grid_lon.reshape((1, ngrid))
    grid_lat = grid_lat.reshape((1, ngrid))

    origin = np.average(pc, axis=0)  # the center of the unit sphere
    pc -= origin  # for looking from the perspective of the origin
    npc = len(pc)

    pc_x, pc_y, pc_z = pc[:, 0], pc[:, 1], pc[:, 2]

    pc_r = np.sqrt(pc_x * pc_x + pc_y * pc_y + pc_z * pc_z)
    pc_lat = np.arcsin(pc_z / pc_r)
    pc_lon = np.arctan2(pc_y, pc_x)

    pc_r = pc_r.reshape((npc, 1))
    pc_lat = pc_lat.reshape((npc, 1))
    pc_lon = pc_lon.reshape((npc, 1))

    dist = -np.cos(grid_lat) * np.cos(pc_lat) * np.cos(grid_lon - pc_lon) + np.sin(grid_lat) * np.sin(pc_lat)

    argmin = np.argmin(dist, axis=0)
    grid_r = pc_r[argmin].reshape((nlat, nlon))
    grid.data = grid_r  # data of the projection onto the unit sphere

    argmin = np.argmin(dist, axis=1)  # argmin on a different axis
    flag = np.zeros(ngrid, dtype=bool)
    flag[argmin] = True  # indicates the polar angles for which the grid data can be interpreted as a point

    return grid, flag, origin


def convert_grid_to_pc(grid, flag, origin):
    """ Function for reconstructing a point cloud from its projection onto the unit sphere. """

    nlon = grid.nlon
    nlat = grid.nlat
    lon = np.linspace(0, nlon * np.pi * 2 / (nlon - 1), num=nlon, endpoint=False)
    lat = np.linspace(nlat // 2 * np.pi / -(nlat - 1), np.pi / 2, num=nlat, endpoint=True)
    lon = np.broadcast_to(lon.reshape((1, nlon)), (nlat, nlon))
    lat = np.broadcast_to(lat.reshape((nlat, 1)), (nlat, nlon))
    r = grid.data

    z = np.sin(lat) * r
    t = np.cos(lat) * r
    x = t * np.cos(lon)
    y = t * np.sin(lon)

    pc = np.zeros(grid.data.shape + (3,))
    pc[:, :, 0] = x
    pc[:, :, 1] = y
    pc[:, :, 2] = -z  # must have the minus
    pc = pc.reshape((-1, 3))
    pc = pc[flag, :]  # only the flagged polar angles must be used in the point cloud reconstruction
    pc += origin  # translate to the original origin

    return pc


def low_pass_filter(grid, sigma):
    ''' Function for diminishing high frequency components in the spherical harmonics representation. '''

    # transform to the frequency domain
    clm = grid.expand()

    # create filter weights
    weights = getGaussianKernel(clm.coeffs.shape[1] * 2 - 1, sigma)[clm.coeffs.shape[1] - 1:]
    weights /= weights[0]

    # low-pass filtering
    clm.coeffs *= weights

    # transform back into spatial domain
    low_passed_grid = clm.expand()

    return low_passed_grid


def duplicate_randomly(pc, size):
    """ Make up for the point loss due to conflictions in the projection process. """
    loss_cnt = size - pc.shape[0]
    if loss_cnt <= 0:
        return pc
    rand_indices = np.random.randint(0, pc.shape[0], size=loss_cnt)
    dup = pc[rand_indices]
    return np.concatenate((pc, dup))


def our_method(pc, lmax, sigma, pc_size=1024, device="cuda"):
    grid, flag, origin = convert_pc_to_grid(pc, lmax, device)
    smooth_grid = low_pass_filter(grid, sigma)
    smooth_pc = convert_grid_to_pc(smooth_grid, flag, origin)
    smooth_pc = duplicate_randomly(smooth_pc, pc_size)
    return smooth_pc

import open3d as o3d
import os
import sys
root_path=os.path.abspath(os.path.join(os.getcwd()))
root_path_2=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root_path_2))
sys.path.append(os.path.join(root_path_2,'baselines'))
from baselines.dataset import ModelNet40Attack, ModelNet40
import argparse
from torch.utils.data import DataLoader
import visdom
import tqdm
# vis = visdom.Visdom(port=8097)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='/home/jqf/桌面/benchmark_pc_attack1-master（复件）/baselines/attack_scripts/results/mn40_1024/SIA/SIA-pointnet-success_1.0000.npz')
    parser.add_argument('--batch_size', type=int, default=1, metavar='BS',
                        help='Size of batch')
    args = parser.parse_args()


    # data_root='/home/jqf/桌面/benchmark_pc_attack1-master（复件）/baselines/attack_scripts/attack/results/mn40_1024/kNN/kNN-pointnet-logits_kappa=15.0-success_0.8545-rank_-1.npz'
    test_set = ModelNet40(args.data_root, num_points=1024,
                          normalize=False, partition='test',
                          augmentation=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=0,
                             pin_memory=True, drop_last=False)
    sub_roots = args.data_root.split('/')
    filename = sub_roots[-1]
    data_folder =args.data_root[:args.data_root.rindex(filename)]

    # data to defend
    batch_size = 1
    npz_data = np.load(args.data_root)
    test_pc = npz_data['test_pc']
    test_label = npz_data['test_label']
    target_label = npz_data['target_label']
    all_d_pc=[]
    all_real_label=[]
    all_defend_pc = []
    for batch_idx in tqdm.trange(0, len(test_pc), batch_size):
        batch_pc = test_pc[batch_idx:batch_idx + batch_size]
        batch_pc = torch.from_numpy(batch_pc)[..., :3]
        batch_pc = batch_pc.float().cuda()
        # raw_point=test_set.data[100,:,:] #choose point cloud
        defense_pc=our_method(batch_pc.squeeze(0),20,20)
        defend_batch_pc=defense_pc[np.newaxis,:]
        # p_color = torch.ones(defense_pc.shape[0])
        # plot_pc = defense_pc[:, :]
        # # plot_pc = plot_pc.transpose(1, 0)
        # vis.scatter(X=plot_pc[:, torch.LongTensor([2, 0, 1])], Y=p_color, win=2,
        #             opts={'title': "Generated Pointcloud", 'markersize': 3, 'webgl': True})
        # save results
        if isinstance(defend_batch_pc, list) or \
                isinstance(defend_batch_pc, tuple):
            defend_batch_pc = [
                pc.detach().cpu().numpy().astype(np.float32) for
                pc in defend_batch_pc
            ]
        else:
            defend_batch_pc = defend_batch_pc.astype(np.float32)
            defend_batch_pc = [pc for pc in defend_batch_pc]

        all_defend_pc += defend_batch_pc

    all_defend_pc = np.array(all_defend_pc)
    np.savez('LPF.npz',
             test_pc=all_defend_pc,
             test_label=test_label.astype(np.uint8),
             target_label=target_label.astype(np.uint8))

