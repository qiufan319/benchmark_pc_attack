import os
import time
from tqdm import tqdm
import argparse
import numpy as np

import torch

from torch.utils.data import DataLoader


import sys
root_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from config import BEST_WEIGHTS
from config import MAX_FGM_PERTURB_BATCH as BATCH_SIZE
from dataset import ModelNetDataLoader, ModelNet40Attack
from model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg,pointnet_cls,GDANET,RPC
from util.utils import str2bool, set_seed
from attack import CrossEntropyAdvLoss, LogitsAdvLoss,UntargetedLogitsAdvLoss
from attack import ChamferDist, L2Dist
from attack import ClipPointsLinf
from attack.Advpc.latent_3d_points.src import encoders_decoders
from attack import CWAdvPC

def attack():
    model.eval()
    all_adv_pc = []
    all_real_lbl = []
    all_target_lbl = []
    all_ori_pc = []
    num = 0
    for pc, label, target in tqdm(test_loader):
        with torch.no_grad():
            pc, label = pc.float().cuda(non_blocking=True), \
                label.long().cuda(non_blocking=True)
            target_label = target.long().cuda(non_blocking=True)

        # attack!
        _, best_pc, success_num = attacker.attack(pc, target_label, label)

        # results
        num += success_num
        all_adv_pc.append(best_pc)
        all_real_lbl.append(label.detach().cpu().numpy())
        all_target_lbl.append(target_label.detach().cpu().numpy())
        all_ori_pc.append(pc.detach().cpu().numpy())
    # accumulate results
    all_adv_pc = np.concatenate(all_adv_pc, axis=0)  # [num_data, K, 3]
    all_real_lbl = np.concatenate(all_real_lbl, axis=0)  # [num_data]
    all_target_lbl = np.concatenate(all_target_lbl, axis=0)  # [num_data]
    all_ori_pc = np.concatenate(all_ori_pc, axis=0)
    return all_adv_pc, all_real_lbl, all_target_lbl, num ,all_ori_pc


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='baselines/data/attack_data.npz')
    parser.add_argument('--attack_data_root', type=str,
                        default='baselines/data/attack_data.npz')
    parser.add_argument('--model', type=str, default='pointnet2',
                        choices=['pointnet', 'pointnet2',
                                 'dgcnn', 'pointconv', 'curvenet', 'pct', 'gda', 'pointcnn', 'rpc'])
    parser.add_argument('--ae_model_path', type=str,
                        default='baselines/attack/Advpc/latent_3d_points/src/logs/mn40/AE/2021-12-31 15:15:52_1024/BEST_model9800_CD_0.0038.pth')
    parser.add_argument('--feature_transform', type=str2bool, default=False,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--dataset', type=str, default='mn40', metavar='N',
                        choices=['mn40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='BS',
                        help='Size of batch')
    parser.add_argument('--num_point', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    parser.add_argument('--adv_func', type=str, default='logits',
                        choices=['logits', 'cross_entropy'],
                        help='Adversarial loss function to use')
    parser.add_argument('--kappa', type=float, default=30.,
                        help='min margin in logits adv loss')
    parser.add_argument('--budget', type=float, default=0.18,
                        help='FGM attack budget')
    parser.add_argument('--GAMMA', type=float, default=0.25,
                        help='hyperparameter gamma')
    parser.add_argument('--binary_step', type=int, default=2, metavar='N',
                        help='Number of binary search step')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--attack_lr', type=float, default=1e-2,
                        help='lr in attack training optimization')
    parser.add_argument('--num_iter', type=int, default=200, metavar='N',
                        help='Number of iterations in each search step')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')

    args = parser.parse_args()
    BATCH_SIZE = BATCH_SIZE[args.num_point]
    BEST_WEIGHTS = BEST_WEIGHTS[args.dataset][args.num_point]
    if args.batch_size == -1:
        args.batch_size = BATCH_SIZE[args.model]
    set_seed(1)
    print(args)



    # build model
    import importlib

    if args.model.lower() == 'dgcnn':
        model = DGCNN(args.emb_dims, args.k, output_channels=40)
    elif args.model.lower() == 'pointnet':
        model = PointNetCls(k=40, feature_transform=args.feature_transform)
    elif args.model.lower() == 'pointnet2':
        model = PointNet2ClsSsg(num_classes=40)
    elif args.model.lower() == 'pointconv':
        model = PointConvDensityClsSsg(num_classes=40)
    elif args.model.lower() == 'curvenet':
        model_tmp = importlib.import_module('model.SIA.' + args.model)
        model = model_tmp.get_model()
        model = model.cuda()
    elif args.model.lower() == 'pct':
        model_tmp = importlib.import_module('model.SIA.' + args.model)
        model = model_tmp.get_model()
        model = model.cuda()
    elif args.model.lower() == 'gda':
        model_tmp = GDANET()
        model = model_tmp.cuda()
    elif args.model.lower() == 'rpc':
        model_tmp = RPC(args)
        model = model_tmp.cuda()

    else:
        print('Model not recognized')
        exit(-1)

    state_dict = torch.load(
        BEST_WEIGHTS[args.model], map_location='cpu')
    print('Loading weight {}'.format(BEST_WEIGHTS[args.model]))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # eliminate 'module.' in keys
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    model = model.cuda()

    #AutoEncoder model
    ae_model = encoders_decoders.AutoEncoder(3)
    ae_state_dict = torch.load(args.ae_model_path)
    print('Loading ae weight {}'.format(args.ae_model_path))
    try:
        ae_model.load_state_dict(ae_state_dict)
    except RuntimeError:
        ae_state_dict = {k[7:]: v for k, v in ae_state_dict.items()}
        ae_model.load_state_dict(ae_state_dict)

    ae_model = ae_model.cuda()


    # setup attack settings
    if args.adv_func == 'logits':
        adv_func = UntargetedLogitsAdvLoss(kappa=args.kappa)
    else:
        adv_func = CrossEntropyAdvLoss()
    clip_func = ClipPointsLinf(budget=args.budget)
    dist_func = ChamferDist()

    # attack
    test_set = ModelNet40Attack(args.data_root, num_points=args.num_points,
                                normalize=True)
    # test_sampler = DistributedSampler(test_set, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=4,
                             pin_memory=True, drop_last=False,
                             )


    attacker = CWAdvPC(model, ae_model, adv_func, dist_func,
                         attack_lr=args.attack_lr,
                         binary_step=args.binary_step,
                         num_iter=args.num_iter, GAMMA=args.GAMMA,
                         clip_func=clip_func)

    # run attack
    attacked_data, real_label, target_label, success_num,ori_pc = attack()

    # accumulate results
    data_num = len(test_set)
    success_rate = float(success_num) / float(data_num)

    # save results
    save_path = 'baselines/result'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = 'AdvPC.npz'
    np.savez(os.path.join(save_path, save_name),
             test_pc=attacked_data.astype(np.float32),
             test_label=real_label.astype(np.uint8),
             target_label=target_label.astype(np.uint8),
             ori_pc=ori_pc.astype(np.float32))