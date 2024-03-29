"""Targeted FGM variants attack."""

import os
from tqdm import tqdm
import argparse
import numpy as np

import torch
torch.manual_seed(1)
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

import sys
import os
root_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from config import BEST_WEIGHTS
from config import MAX_FGM_PERTURB_BATCH as BATCH_SIZE
from dataset import ModelNet40Attack
from model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg,pointnet_cls,GDANET,RPC
from util.utils import str2bool, set_seed
from attack import FGM, IFGM, MIFGM, PGD
from attack import CrossEntropyAdvLoss, LogitsAdvLoss,UntargetedLogitsAdvLoss
from attack import ClipPointsL2
import random

def target_lable_generate(x,num_class):
    target_label=random.sample(range(0,num_class),len(x))
    for i in range(len(x)):
        if x[i]==target_label[i]:
            if target_label[i]==(num_class-1):
                target_label[i]=0
            else:
                target_label[i]+=1
    return target_label

def attack():
    model.eval()
    all_adv_pc = []
    all_real_lbl = []
    all_target_lbl = []
    all_pc=[]
    num = 0
    for pc, label,_ in tqdm(test_loader):
        with torch.no_grad():
            pc, label = pc.float().cuda(non_blocking=True), \
                label.long().cuda(non_blocking=True)
            # target_label = target_lable_generate(label,40).long().cuda(non_blocking=True)
        o=model(pc.permute(0, 2, 1).contiguous())
        if o.size(0)>1:
            y_pred, _, _ = o
        else:
            y_pred=o
        # attack!
        best_pc, success_num = attacker.attack(pc, label)
        # results
        num += success_num
        all_adv_pc.append(best_pc)
        all_real_lbl.append(label.detach().cpu().numpy())
        all_target_lbl.append(label.detach().cpu().numpy())
        all_pc.append(pc.detach().cpu().numpy())
    # accumulate results
    all_adv_pc = np.concatenate(all_adv_pc, axis=0)  # [num_data, K, 3]
    all_real_lbl = np.concatenate(all_real_lbl, axis=0)  # [num_data]
    all_target_lbl = np.concatenate(all_target_lbl, axis=0)  # [num_data]
    all_pc=np.concatenate(all_pc)
    return all_adv_pc, all_real_lbl, all_target_lbl, num ,all_pc


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='baselines/data/attack_data.npz')
    parser.add_argument('--model', type=str, default='pointnet2',
                        choices=['pointnet', 'pointnet2',
                                 'dgcnn', 'pointconv', 'curvenet', 'pct', 'gda', 'pointcnn', 'rpc'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv,curvenet,pct,]')
    parser.add_argument('--feature_transform', type=str2bool, default=False,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--dataset', type=str, default='mn40', metavar='N',
                        choices=['mn40', 'remesh_mn40', 'opt_mn40', 'conv_opt_mn40'])
    parser.add_argument('--batch_size', type=int, default=1, metavar='BS',
                        help='Size of batch')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    parser.add_argument('--adv_func', type=str, default='logits',
                        choices=['logits', 'cross_entropy'],
                        help='Adversarial loss function to use')
    parser.add_argument('--kappa', type=float, default=0.,
                        help='min margin in logits adv loss')
    parser.add_argument('--attack_type', type=str, default='pgd', metavar='N',
                        help='Attack method to use')
    parser.add_argument('--budget', type=float, default=0.08,
                        help='FGM attack budget')
    parser.add_argument('--num_iter', type=int, default=50,
                        help='IFGM iterate step')
    parser.add_argument('--mu', type=float, default=1.,
                        help='momentum factor for MIFGM attack')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    args = parser.parse_args()
    BATCH_SIZE = BATCH_SIZE[args.num_points]
    BEST_WEIGHTS = BEST_WEIGHTS[args.dataset][args.num_points]
    if args.batch_size == -1:
        args.batch_size = BATCH_SIZE[args.model]
    set_seed(1)
    print(args)

    #dist.init_process_group(backend='nccl')
    #torch.cuda.set_device(args.local_rank)
    cudnn.benchmark = True

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

    # load model weight
    state_dict = torch.load(
        BEST_WEIGHTS[args.model], map_location='cpu')
    print('Loading weight {}'.format(BEST_WEIGHTS[args.model]))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # eliminate 'module.' in keys
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    # distributed mode on multiple GPUs!
    # much faster than nn.DataParallel
    # model = DistributedDataParallel(
    #     model.cuda(), device_ids=[args.local_rank])

    # setup attack settings
    # budget, step_size, number of iteration
    # settings adopted from CVPR'20 paper GvG-P
    delta = args.budget
    args.budget = args.budget * \
        np.sqrt(args.num_points * 3)  # \delta * \sqrt(N * d)
    args.num_iter = int(args.num_iter)
    args.step_size = args.budget / float(args.num_iter)
    # which adv_func to use?
    if args.adv_func == 'logits':
        adv_func = UntargetedLogitsAdvLoss(kappa=args.kappa)
    else:
        adv_func = CrossEntropyAdvLoss()
    clip_func = ClipPointsL2(budget=args.budget)

    # 4 variants of FGM
    if args.attack_type.lower() == 'fgm':
        attacker = FGM(model, adv_func=adv_func,
                       budget=args.budget, dist_metric='l2')
    elif args.attack_type.lower() == 'ifgm':
        attacker = IFGM(model, adv_func=adv_func,
                        clip_func=clip_func, budget=args.budget, step_size=args.step_size,
                        num_iter=args.num_iter, dist_metric='l2')
    elif args.attack_type.lower() == 'mifgm':
        attacker = MIFGM(model, adv_func=adv_func,
                         clip_func=clip_func, budget=args.budget, step_size=args.step_size,
                         num_iter=args.num_iter, mu=args.mu, dist_metric='l2')
    elif args.attack_type.lower() == 'pgd':
        attacker = PGD(model, adv_func=adv_func,
                       clip_func=clip_func, budget=args.budget, step_size=args.step_size,
                       num_iter=args.num_iter, dist_metric='l2')

    # attack
    test_set = ModelNet40Attack(args.data_root, num_points=args.num_points,
                                normalize=True)
    #test_sampler = DistributedSampler(test_set, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=0,
                             pin_memory=True, drop_last=False,
                             sampler=None)


    # run attack
    attacked_data, real_label, target_label, success_num ,all_pc= attack()

    # accumulate results
    data_num = len(test_set)
    success_rate = float(success_num) / float(data_num)

    # save results
    save_path = 'baselines/result'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = '{}.npz'.\
        format(args.attack_type.lower())
    np.savez(os.path.join(save_path, save_name),
             test_pc=attacked_data.astype(np.float32),
             test_label=real_label.astype(np.uint8),
             target_label=target_label.astype(np.uint8),
             ori_pc=all_pc.astype(np.float32))
