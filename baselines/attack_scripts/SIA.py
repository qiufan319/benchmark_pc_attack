# -*- coding: utf-8 -*-

import os
import sys
root_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
import argparse
import time
from util.utils import str2bool, set_seed
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
from model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg,pointnet_cls
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import BEST_WEIGHTS
from attack.SIA.attacks import PointCloudAttack
import model
#from data_utils.ModelNetDataLoader import ModelNetDataLoader
#from data_utils.ShapeNetDataLoader import PartNormalDataset
from torch.utils.data import DataLoader, TensorDataset
from dataset import ModelNet40Attack

#from util.logging import Logging_str
from util.utils import set_seed

from attack.SIA.attacks import PointCloudAttack
from util.set_distance_SIA import ChamferDistance, HausdorffDistance



# def load_data(args):
#     """Load the dataset from the given path.
#     """
#     print('Start Loading Dataset...')
#     if args.dataset == 'ModelNet40':
#         TEST_DATASET = ModelNetDataLoader(
#             root=args.data_path,
#             npoint=args.input_point_nums,
#             split='test',
#             normal_channel=True
#         )
#     elif args.dataset == 'ShapeNetPart':
#         TEST_DATASET = PartNormalDataset(
#             root=args.data_path,
#             npoints=args.input_point_nums,
#             split='test',
#             normal_channel=True
#         )
#     else:
#         raise NotImplementedError
#
#     testDataLoader = torch.utils.data.DataLoader(
#         TEST_DATASET,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers
#     )
#     print('Finish Loading Dataset...')
#     return testDataLoader



def data_preprocess(data):
    """Preprocess the given data and label.
    """
    points= data[0]
    target=data[1]
    points = points # [B, N, C]
    #target = target[:, 0] # [B]

    points = points.cuda()
    target = target.cuda()

    return points, target


def save_tensor_as_txt(points, filename):
    """Save the torch tensor into a txt file.
    """
    points = points.squeeze(0).detach().cpu().numpy()
    with open(filename, "a") as file_object:
        for i in range(points.shape[0]):
            # msg = str(points[i][0]) + ' ' + str(points[i][1]) + ' ' + str(points[i][2])
            msg = str(points[i][0]) + ' ' + str(points[i][1]) + ' ' + str(points[i][2]) + \
                ' ' + str(points[i][3].item()) +' ' + str(points[i][3].item()) + ' '+ str(1-points[i][3].item())
            file_object.write(msg+'\n')
        file_object.close()
    print('Have saved the tensor into {}'.format(filename))


def main():
    # load data
    # test_loader = load_data(args)
    test_set = ModelNet40Attack(args.data_path, num_points=args.input_point_nums,
                                normalize=True)
    # test_sampler = DistributedSampler(test_set, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=0,
                             pin_memory=True, drop_last=False)
    num_class = 0
    if args.dataset == 'mn40':
        num_class = 40
    elif args.dataset == 'ShapeNetPart':
        num_class = 16
    assert num_class != 0
    args.num_class = num_class

    # load model
    # attack = PointCloudAttack(args)
    # build model
    if args.surrogate_model.lower() == 'dgcnn':
        model = DGCNN(args.emb_dims, args.k, output_channels=40).cuda()
    elif args.surrogate_model.lower() == 'pointnet':
        model = PointNetCls(k=40, feature_transform=args.feature_transform).cuda()
    elif args.surrogate_model.lower() == 'pointnet2':
        model = PointNet2ClsSsg(num_classes=40)
    elif args.surrogate_model.lower() == 'pointconv':
        model = PointConvDensityClsSsg(num_classes=40).cuda()
    elif args.surrogate_model.lower() == 'pointcloud_cls':
        model = pointnet_cls.cuda()
    else:
        print('Model not recognized')
        exit(-1)

    # load model weight
    state_dict = torch.load(
        BEST_WEIGHTS[args.surrogate_model], map_location='cpu')
    print('Loading weight {}'.format(BEST_WEIGHTS[args.surrogate_model]))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # eliminate 'module.' in keys
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    # start attack
    atk_success = 0
    avg_query_costs = 0.
    avg_mse_dist = 0.
    avg_chamfer_dist = 0.
    avg_hausdorff_dist = 0.
    avg_time_cost = 0.
    chamfer_loss = ChamferDistance()
    hausdorff_loss = HausdorffDistance()
    pc=[]
    label=[]
    target_label=[]
    num = len(test_loader)
    l = [None] * num
    for batch_id, data in tqdm(enumerate(test_loader),colour='WHITE',total=len(l),ncols =200,file=sys.stdout):
        # prepare data for testing
        points, target = data_preprocess(data)
        target = target.long()

        # start attack
        t0 = time.perf_counter()
        attack = PointCloudAttack(args,model)
        adv_points, adv_target, query_costs = PointCloudAttack.run(attack,points,target)
        t1 = time.perf_counter()
        avg_time_cost += t1 - t0
        pc.append(adv_points.cpu().numpy())
        label.append(target.detach().cpu().numpy())
        target_label.append(adv_target.detach().cpu().numpy())
        if not args.query_attack_method is None:
            print('>>>>>>>>>>>>>>>>>>>>>>>')
            print('Query cost: ', query_costs)
            print('>>>>>>>>>>>>>>>>>>>>>>>')
            avg_query_costs += query_costs
        atk_success += 1 if adv_target != target else 0

        # modified point num count
        points = points[:,:,:3].data # P, [1, N, 3]
        pert_pos = torch.where(abs(adv_points-points).sum(2))
        count_map = torch.zeros_like(points.sum(2))
        count_map[pert_pos] = 1.
        # print('Perturbed point num:', torch.sum(count_map).item())

        avg_mse_dist += np.sqrt(F.mse_loss(adv_points, points).detach().cpu().numpy() * 3072)
        avg_chamfer_dist += chamfer_loss(adv_points, points)
        avg_hausdorff_dist += hausdorff_loss(adv_points, points)
        if batch_id%100==0:
            success_result=atk_success / (batch_id + 1)
            # print('Attack success rate: ', success_result)
            print('\nAttack number {}, success {}/{}\n'
                  'adv_loss: {:.4f}, dist_loss: {:.4f}'.
                  format((batch_id+1), atk_success, (batch_id + 1),
                         avg_mse_dist.item(), (avg_chamfer_dist+avg_hausdorff_dist).item()))
    atk_success /= batch_id + 1
    print('Attack success rate: ', atk_success)
    avg_time_cost /= batch_id + 1
    print('Average time cost: ', avg_time_cost)
    if not args.query_attack_method is None:
        avg_query_costs /= batch_id + 1
        print('Average query cost: ', avg_query_costs)
    avg_mse_dist /= batch_id + 1
    print('Average MSE Dist:', avg_mse_dist)
    avg_chamfer_dist /= batch_id + 1
    print('Average Chamfer Dist:', avg_chamfer_dist.item())
    avg_hausdorff_dist /= batch_id + 1
    print('Average Hausdorff Dist:', avg_hausdorff_dist.item())
    all_adv_pc = np.concatenate(pc, axis=0)  # [num_data, K, 3]
    all_real_lbl = np.concatenate(label, axis=0)  # [num_data]
    all_target_lbl = np.concatenate(target_label, axis=0)
    save_path = 'attack_scripts/attack/results/{}_{}/SIA'. \
        format(args.dataset, args.input_point_nums)
    save_name = 'SIA-{}-success_{:.4f}.npz'. \
        format(args.target_model,
               atk_success)
    np.savez(os.path.join(save_path, save_name),
             test_pc=all_adv_pc.astype(np.float32),
             test_label=all_real_lbl.astype(np.uint8),
             target_label=all_target_lbl.astype(np.uint8))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Shape-invariant 3D Adversarial Point Clouds')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    parser.add_argument('--input_point_nums', type=int, default=1024,
                        help='Point nums of each point cloud')
    parser.add_argument('--seed', type=int, default=2022, metavar='S',
                        help='random seed (default: 2022)')
    parser.add_argument('--dataset', type=str, default='mn40',
                        choices=['mn40', 'ShapeNetPart'])
    parser.add_argument('--data_path', type=str, 
                        default='baselines/data/attack_data.npz')
    parser.add_argument('--normal', action='store_true', default=False,
                        help='Whether to use normal information [default: False]')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Worker nums of data loading.')
    parser.add_argument('--feature_transform', type=str2bool, default=False,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--transfer_attack_method', type=str, default='ifgm_ours',
                        choices=['ifgm_ours'])
    parser.add_argument('--query_attack_method', type=str, default=None,
                        choices=['simbapp', 'simba', 'ours'])
    parser.add_argument('--surrogate_model', type=str, default='pointnet',
                        choices=['pointnet', 'pointnet2', 'dgcnn', 'pointconv', 'pointcnn', 'paconv', 'pct', 'curvenet', 'simple_view'])
    parser.add_argument('--target_model', type=str, default='pointnet',
                        choices=['pointnet', 'pointnet2', 'dgcnn', 'pointconv', 'pointcnn', 'paconv', 'pct', 'curvenet', 'simple_view'])
    parser.add_argument('--defense_method', type=str, default=None,
                        choices=['sor', 'srs', 'dupnet'])
    parser.add_argument('--top5_attack', action='store_true', default=False,
                        help='Whether to attack the top-5 prediction [default: False]')

    parser.add_argument('--max_steps', default=50, type=int,
                        help='max iterations for black-box attack')
    parser.add_argument('--eps', default=0.16, type=float,
                        help='epsilon of perturbation')
    parser.add_argument('--step_size', default=0.07, type=float,
                        help='step-size of perturbation')
    args = parser.parse_args()

    # basic configuration
    set_seed(args.seed)
    BEST_WEIGHTS = BEST_WEIGHTS[args.dataset][args.input_point_nums]
    args.device = torch.device("cuda")
    import importlib
    # main loop
    main()
