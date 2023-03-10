from __future__ import absolute_import, division, print_function
from tqdm import tqdm
import sys
import os
root_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
import argparse
import math
import os
import sys
import time
from torch.utils.data import DataLoader
from dataset import ModelNet40Attack,ModelNet40
import numpy as np
import scipy.io as sio
from config import BEST_WEIGHTS
import torch
from util.utils import str2bool, set_seed
import torch.nn as nn
import torch.optim as optim
from pytorch3d.io import load_obj, save_obj
from model import DGCNN, PointNetCls, feature_transform_reguliarzer, \
    PointNet2ClsSsg, PointConvDensityClsSsg, RPC, GDANET
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from torch.autograd import Variable

from attack.GAO.Attacker import geoA3_attack
from attack.GAO.Lib.utility import (Average_meter, Count_converge_iter, Count_loss_iter,
                         _compare, accuracy, estimate_normal_via_ori_normal,
                         farthest_points_sample)
from dataset import ModelNet40_Geo
from attack import CrossEntropyAdvLoss, LogitsAdvLoss

def main(cfg):

    if cfg.attack == 'GeoA3_mesh':
        assert False, 'Not uploaded yet.'

    if cfg.attack_label == 'Untarget':
        targeted = False
    else:
        targeted = True

    # print('=>Creating dir')
    saved_root = os.path.join('attack_scripts/result', cfg.model + '_num_points' + str(cfg.num_points))

    if cfg.attack == 'GeoA3' or cfg.attack == 'GeoA3_mesh':
        saved_dir = str(cfg.attack) + '_' +  str(cfg.id) +  '_BiStep' + str(cfg.binary_max_steps) + '_IterStep' + str(cfg.iter_max_steps) + '_Opt' + cfg.optim  +  '_Lr' + str(cfg.lr) + '_Initcons' + str(cfg.initial_const) + '_' + cfg.cls_loss_type + '_' + str(cfg.dis_loss_type) + 'Loss' + str(cfg.dis_loss_weight)

        if cfg.hd_loss_weight != 0:
            saved_dir = saved_dir + '_HDLoss' + str(cfg.hd_loss_weight)

        if cfg.curv_loss_weight != 0:
            saved_dir = saved_dir + '_CurLoss' + str(cfg.curv_loss_weight) + '_k' + str(cfg.curv_loss_knn)

        if cfg.uniform_loss_weight != 0:
            saved_dir = saved_dir + '_UniLoss' + str(cfg.uniform_loss_weight)

        if cfg.laplacian_loss_weight != 0:
            saved_dir = saved_dir + '_LapLoss' + str(cfg.laplacian_loss_weight)

        if cfg.edge_loss_weight != 0:
            saved_dir = saved_dir + '_EdgeLoss' + str(cfg.edge_loss_weight)

        if cfg.is_partial_var:
            saved_dir = saved_dir + '_PartOpt' + '_k' + str(cfg.knn_range)

        if cfg.is_use_lr_scheduler:
            saved_dir = saved_dir + '_LRExp'

        if cfg.is_pro_grad:
            saved_dir = saved_dir + '_ProGrad'
            if cfg.is_real_offset:
                saved_dir = saved_dir + 'RO'

        if cfg.cc_linf != 0:
            saved_dir = saved_dir + '_cclinf' + str(cfg.cc_linf)


        if cfg.is_pre_jitter_input:
            saved_dir = saved_dir + '_PreJitter' + str(cfg.jitter_sigma) + '_' + str(cfg.jitter_clip)
            if cfg.is_previous_jitter_input:
                saved_dir = saved_dir + '_PreviousMethod'
            else:
                saved_dir = saved_dir + '_estNormalVery' + str(cfg.calculate_project_jitter_noise_iter)

    else:
        assert cfg.attack == None
        saved_dir = 'Evaluating_' + str(cfg.id)

    saved_dir = os.path.join(saved_root, cfg.attack_label, saved_dir)
    # print('==>Successfully created {}'.format(saved_dir))

    if cfg.attack == 'GeoA3_mesh':
        trg_dir = os.path.join(saved_dir, 'Mesh')

    if cfg.id == 0:
        seed = 0
    else:
        seed = int(time.time())
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    if (str(cfg.num_points) in cfg.data_dir_file):
        resample_num = -1
    else:
        # FIXME
        #resample_num = cfg.npoint
        resample_num = -1


    test_dataset = ModelNet40_Geo(data_mat_file=cfg.data_dir_file, attack_label=cfg.attack_label, resample_num=resample_num)

    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=0,
                             pin_memory=True, drop_last=False)



    # model
    # print('=>Loading model')
    print('=>Loading model')
    import importlib
    if cfg.model.lower() == 'dgcnn':
        model = DGCNN(cfg.emb_dims, cfg.k, output_channels=40)
    elif cfg.model.lower() == 'pointnet':
        model = PointNetCls(k=40, feature_transform=cfg.feature_transform)
    elif cfg.model.lower() == 'pointnet2':
        model = PointNet2ClsSsg(num_classes=40)
    elif cfg.model.lower() == 'pointconv':
        model = PointConvDensityClsSsg(num_classes=40)
    elif cfg.model.lower() == 'curvenet':
        model_tmp = importlib.import_module('model.SIA.' + cfg.model)
        model = model_tmp.get_model()
        model = model.cuda()
    elif cfg.model.lower() == 'pct':
        model_tmp = importlib.import_module('model.SIA.' + cfg.model)
        model = model_tmp.get_model()
        model = model.cuda()
    elif cfg.model.lower() == 'gda':
        model_tmp = GDANET()
        model = model_tmp.cuda()
    elif cfg.model.lower() == 'rpc':
        model_tmp = RPC(cfg)
        model = model_tmp.cuda()

    # load model weight
    state_dict = torch.load(
        BEST_WEIGHTS[cfg.model], map_location='cpu')
    print('Loading weight {}'.format(BEST_WEIGHTS[cfg.model]))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # eliminate 'module.' in keys
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    model=model.cuda()
    model.eval()
    # print('==>Successfully load pretrained-model from {}'.format(model_path))

    # recording settings
    if cfg.is_record_converged_steps:
        cci = Count_converge_iter(os.path.join(saved_dir, 'Records'))
    if cfg.is_record_loss:
        cli = Count_loss_iter(os.path.join(saved_dir, 'Records'))

    test_acc = Average_meter()


    num_attack_success = 0
    cnt_ins = test_dataset.start_index
    cnt_all = 0

    if cfg.attack_label == 'Untarget':
        targeted = False
        num_attack_classes = 1
    elif cfg.attack_label == 'Random':
        targeted = True
        num_attack_classes = 1
    else:
        targeted = True
        num_attack_classes = 9
    all_adv_pc = []
    all_real_lbl = []
    all_target_lbl = []
    all_orig_pc=[]
    num=len(test_loader)
    l = [None] * num

    if cfg.adv_func == 'logits':
        adv_func = LogitsAdvLoss(kappa=cfg.kappa)
    else:
        adv_func = CrossEntropyAdvLoss()
    for i, data in tqdm(enumerate(test_loader),colour='WHITE',total=len(l)):

        if cfg.attack == 'GeoA3':
            pc = data[0]
            normal = data[1]
            gt_labels = data[2]
            if pc.size(3) == 3:
                pc = pc.permute(0,1,3,2)
            if normal.size(3) == 3:
                normal = normal.permute(0,1,3,2)

            bs, l, _, n = pc.size()
            b = bs*l

            pc = pc.view(b, 3, n).cuda()
            normal = normal.view(b, 3, n).cuda()
            gt_target = gt_labels.view(-1).cuda()



        if cfg.attack is None:
            if n == 10000:
                for i in range(b):
                    with torch.no_grad():
                        output = model(pc[i].unsqueeze(0))
                    acc = accuracy(output.data, gt_target[i].data.unsqueeze(0), topk=(1, ))
                    test_acc.update(acc[0][0], 1)
            else:
                with torch.no_grad():
                    output = model(pc)
                acc = accuracy(output.data, gt_target.data, topk=(1, ))
                test_acc.update(acc[0][0], output.size(0))
            print("Prec@1 {top1.avg:.3f}".format(top1=test_acc))

        elif cfg.attack == 'GeoA3':
            adv_pc, targeted_label, attack_success_indicator, best_attack_step, loss = geoA3_attack.attack(model, data, cfg, i, len(test_loader), adv_func,saved_dir)
            eval_num = 1
        # elif cfg.attack == 'GeoA3_mesh':
        #     adv_mesh, targeted_label, attack_success_indicator, best_attack_step, best_score = geoA3_mesh_attack.attack(model, data, cfg, i, len(test_loader), saved_dir)
        #     eval_num = 1
        else:
            assert False, "Wrong type of attack."

        if cfg.attack == 'GeoA3':
            if cfg.is_record_converged_steps:
                cci.record_converge_iter(best_attack_step)
            if cfg.is_record_loss:
                cli.record_loss_iter(loss)

            if cfg.is_save_normal:
                with torch.no_grad():
                    # the loop here is for memory save
                    knn_normal = torch.zeros_like(adv_pc)
                    for idx in range(b):
                        knn_normal[idx] = estimate_normal_via_ori_normal(adv_pc[idx].unsqueeze(0), dense_point[idx].unsqueeze(0), dense_normal[idx].unsqueeze(0), k=3)
                saved_normal = knn_normal.cpu().numpy()

            for _ in range(0,eval_num):
                with torch.no_grad():
                    if adv_pc.size(2) > cfg.num_points:
                        eval_points = farthest_points_sample(adv_pc, cfg.num_points)
                    else:
                        eval_points = adv_pc
                    test_adv_output,_,_ = model(eval_points)
                attack_success_iter = _compare(torch.max(test_adv_output,1)[1].data, targeted_label, gt_target.cuda(), targeted)
                try:
                    attack_success += attack_success_iter
                except:
                    attack_success = attack_success_iter

            saved_pc = adv_pc.cpu().clone().numpy()
            saved_ori_pc = pc.cpu().clone().numpy()
            for k in range(b):
                if attack_success_indicator[k].item():
                    num_attack_success += 1
                    real_label=gt_target[k].item()
                    a_pc=saved_pc[k]
                    ori_pc=saved_ori_pc[k]
                    attack_label = torch.max(test_adv_output,1)[1].data[k].item()
                    all_adv_pc.append(a_pc)
                    all_real_lbl.append(real_label)
                    all_target_lbl.append(attack_label)
                    all_orig_pc.append(ori_pc)
    all_adv_pc_2 = np.array(all_adv_pc)
    all_adv_pc_2=all_adv_pc_2.transpose(0,2,1)
    all_real_lbl_2 = np.array(all_real_lbl)  # [num_data]
    all_target_lbl_2 = np.array(all_target_lbl)
    all_orig_pc=np.asarray(all_orig_pc).transpose(0,2,1)

    save_path = 'baselines/result'
    isExists = os.path.exists(save_path)
    if not isExists:
        os.makedirs(save_path)
    np.savez(os.path.join(save_path,'GeoA3.npz'),
             test_pc=all_adv_pc_2.astype(np.float32),
             test_label=all_real_lbl_2.astype(np.float32),
             target_label=all_target_lbl_2.astype(np.uint8),
             ori_pc=all_orig_pc.astype(np.float32))


    print('Finish!')

    return saved_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Attacking')
    #------------Model-----------------------
    parser.add_argument('--id', type=int, default=0, help='')
    parser.add_argument('--model', type=str, default='pointnet2',
                        choices=['pointnet', 'pointnet2',
                                 'dgcnn', 'pointconv', 'curvenet', 'pct', 'gda', 'pointcnn', 'rpc'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv,curvenet,pct,]')
    parser.add_argument('--feature_transform', type=str2bool, default=False,
                        help='whether to use STN on features in PointNet')
    #------------Dataset-----------------------
    parser.add_argument('--data_dir_file', default='baselines/data/attack_data.npz', type=str, help='')
    parser.add_argument('--dense_data_dir_file', default=None, type=str, help='')
    parser.add_argument('-c', '--classes', default=40, type=int, metavar='N', help='num of classes (default: 40)')
    parser.add_argument('-b', '--batch_size', default=18, type=int, metavar='B', help='batch_size (default: 2)')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dataset', type=str, default='mn40', metavar='N')
    #------------Attack-----------------------
    parser.add_argument('--attack', default='GeoA3', type=str, help='GeoA3')
    parser.add_argument('--attack_label', default='Untarget', type=str, help='[All; ...; Untarget]')
    parser.add_argument('--binary_max_steps', type=int, default=10, help='')
    parser.add_argument('--initial_const', type=float, default=10, help='')
    parser.add_argument('--iter_max_steps',  default=500, type=int, metavar='M', help='max steps')
    parser.add_argument('--optim', default='adam', type=str, help='adam| sgd')
    parser.add_argument('--lr', type=float, default=0.01, help='')
    parser.add_argument('--eval_num', type=int, default=1, help='')
    ## cls loss
    parser.add_argument('--cls_loss_type', default='CE', type=str, help='Margin | CE')
    parser.add_argument('--confidence', type=float, default=0, help='confidence for margin based attack method')
    ## distance loss
    parser.add_argument('--dis_loss_type', default='CD', type=str, help='CD | L2 | None')
    parser.add_argument('--dis_loss_weight', type=float, default=1.0, help='')
    parser.add_argument('--is_cd_single_side', action='store_true', default=False, help='')
    ## hausdorff loss
    parser.add_argument('--hd_loss_weight', type=float, default=0.1, help='')
    ## normal loss
    parser.add_argument('--curv_loss_weight', type=float, default=1.0, help='')
    parser.add_argument('--curv_loss_knn', type=int, default=16, help='')
    ## uniform loss
    parser.add_argument('--uniform_loss_weight', type=float, default=0.0, help='')
    ## KNN smoothing loss
    parser.add_argument('--knn_smoothing_loss_weight', type=float, default=5.0, help='')
    parser.add_argument('--knn_smoothing_k', type=int, default=5, help='')
    parser.add_argument('--knn_threshold_coef', type=float, default=1.10, help='')
    ## Laplacian loss for mesh
    parser.add_argument('--laplacian_loss_weight', type=float, default=0, help='')
    parser.add_argument('--edge_loss_weight', type=float, default=0, help='')
    ## Mesh opt
    parser.add_argument('--is_partial_var', dest='is_partial_var', action='store_true', default=False, help='')
    parser.add_argument('--knn_range', type=int, default=3, help='')
    parser.add_argument('--is_subsample_opt', dest='is_subsample_opt', action='store_true', default=False, help='')
    parser.add_argument('--is_use_lr_scheduler', dest='is_use_lr_scheduler', action='store_true', default=False, help='')
    ## perturbation clip setting
    parser.add_argument('--cc_linf', type=float, default=0.0, help='Coefficient for infinity norm')
    ## Proj offset
    parser.add_argument('--is_real_offset', action='store_true', default=False, help='')
    parser.add_argument('--is_pro_grad', action='store_true', default=False, help='')
    ## Jitter
    parser.add_argument('--is_pre_jitter_input', action='store_true', default=False, help='')
    parser.add_argument('--is_previous_jitter_input', action='store_true', default=False, help='')
    parser.add_argument('--calculate_project_jitter_noise_iter', default=50, type=int,help='')
    parser.add_argument('--jitter_k', type=int, default=16, help='')
    parser.add_argument('--jitter_sigma', type=float, default=0.01, help='')
    parser.add_argument('--jitter_clip', type=float, default=0.05, help='')
    ## PGD-like attack
    parser.add_argument('--step_alpha', type=float, default=5, help='')
    #------------Recording settings-------
    parser.add_argument('--is_record_converged_steps', action='store_true', default=False, help='')
    parser.add_argument('--is_record_loss', action='store_true', default=False, help='')
    #------------OS-----------------------
    parser.add_argument('-j', '--num_workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 0)')
    parser.add_argument('--is_save_normal', action='store_true', default=False, help='')
    parser.add_argument('--is_debug', action='store_true', default=False, help='')
    parser.add_argument('--is_low_memory', action='store_true', default=False, help='')
    parser.add_argument('--adv_func', type=str, default='cross_entropy',
                        choices=['logits', 'cross_entropy'],
                        help='Adversarial loss function to use')
    parser.add_argument('--kappa', type=float, default=0.,
                        help='min margin in logits adv loss')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    cfg  = parser.parse_args()
    print(cfg, '\n')

    BEST_WEIGHTS = BEST_WEIGHTS[cfg.dataset][cfg.num_points]


    main(cfg)

