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
from dataset import ModelNet40Attack
import numpy as np
import scipy.io as sio
from config import BEST_WEIGHTS
import torch
from util.utils import str2bool, set_seed
import torch.nn as nn
import torch.optim as optim
from pytorch3d.io import load_obj, save_obj
from model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from torch.autograd import Variable

from attack.GAO.Attacker import geoA3_attack
from attack.GAO.Lib.utility import (Average_meter, Count_converge_iter, Count_loss_iter,
                         _compare, accuracy, estimate_normal_via_ori_normal,
                         farthest_points_sample)
from dataset import ModelNet40_Geo

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
    # else:
    #     trg_dir = os.path.join(saved_dir, 'PC')
    # if not os.path.exists(trg_dir):
    #     os.makedirs(trg_dir)
    # trg_dir = os.path.join(saved_dir, 'Mat')
    # if not os.path.exists(trg_dir):
    #     os.makedirs(trg_dir)
    # trg_dir = os.path.join(saved_dir, 'Records')
    # if not os.path.exists(trg_dir):
    #     os.makedirs(trg_dir)

    if cfg.id == 0:
        seed = 0
    else:
        seed = int(time.time())
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #data
    # if cfg.attack == 'GeoA3_mesh':
    #     from attack.GAO.Provider.modelnet10_instance250_mesh import \
    #         ModelNet10_250instance_mesh
    #     test_dataset = ModelNet10_250instance_mesh(resume=cfg.data_dir_file, attack_label= cfg.attack_label)
    # else:
    if (str(cfg.num_points) in cfg.data_dir_file):
        resample_num = -1
    else:
        # FIXME
        #resample_num = cfg.npoint
        resample_num = -1

    from attack.GAO.Provider.modelnet10_instance250 import ModelNet40
    # test_dataset = ModelNet40(data_mat_file=cfg.data_dir_file, attack_label=cfg.attack_label, resample_num=resample_num)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers, pin_memory=True)
    # test_size = test_dataset.__len__()
    test_dataset = ModelNet40_Geo(data_mat_file=cfg.data_dir_file, attack_label=cfg.attack_label, resample_num=resample_num)
    # test_sampler = DistributedSampler(test_set, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=0,
                             pin_memory=True, drop_last=False)

    if (cfg.is_save_normal) & (cfg.dense_data_dir_file is not None):
        dense_test_dataset = ModelNet40(data_mat_file=cfg.dense_data_dir_file, attack_label=cfg.attack_label, resample_num=-1)
        dense_test_loader = torch.utils.data.DataLoader(dense_test_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers, pin_memory=True)
        dense_test_size = dense_test_dataset.__len__()
        dense_iter = iter(dense_test_loader)
    else:
        dense_iter = None

    # model
    # print('=>Loading model')
    model_path = os.path.join('pretrain','mn40', str(cfg.model.lower())+'.pth')
    if cfg.model.lower() == 'pointnet':
        model = PointNetCls(k=40, feature_transform=cfg.feature_transform).cuda()
    elif cfg.model.lower() == 'pointnet2':
        model = PointNet2ClsSsg(num_classes=40).cuda()

    # if cfg.arch == 'PointNet':
    #     from model.pointnet import PointNetcls
    #     net = PointNetcls(cfg.classes, npoint=cfg.npoint).cuda()
    # elif cfg.arch == 'PointNetPP':
    #     from model.pointnet2 import PointNet2ClassificationSSG
    #     net = PointNet2ClSSSG(use_xyz=True, use_normal=False).cuda()
    else:
        print('Model not recognized')
        exit(-1)
    # checkpoint = torch.load(model_path)
    # model.load_state_dict(checkpoint['state_dict'])
    # model.eval()

    # load model weight
    state_dict = torch.load(
        BEST_WEIGHTS[cfg.model.lower()], map_location='cpu')
    print('Loading weight {}'.format(BEST_WEIGHTS[cfg.model.lower()]))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # eliminate 'module.' in keys
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    model.eval()
    # print('==>Successfully load pretrained-model from {}'.format(model_path))

    # recording settings
    if cfg.is_record_converged_steps:
        cci = Count_converge_iter(os.path.join(saved_dir, 'Records'))
    if cfg.is_record_loss:
        cli = Count_loss_iter(os.path.join(saved_dir, 'Records'))

    test_acc = Average_meter()
    batch_vertice = []
    batch_faces_idx = []
    batch_gt_label = []

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
    num=len(test_loader)
    l = [None] * num
    for i, data in tqdm(enumerate(test_loader),colour='WHITE',total=len(l)):
        # if cfg.attack == 'GeoA3_mesh':
        #     vertex, _, gt_label = data[0], data[1], data[2]
        #     gt_target = gt_label.view(-1).cuda()
        #     bs, l, _, _ = vertex.size()
        #     b = bs*l
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

            if dense_iter is not None:
                dense_data = dense_iter.next()
                dense_point = dense_data[0]
                dense_normal = dense_data[1]

                if dense_point.size(3) == 3:
                    dense_point = dense_point.permute(0,1,3,2)
                if dense_normal.size(3) == 3:
                    dense_normal = dense_normal.permute(0,1,3,2)

                bs, l, _, n = dense_point.size()
                b = bs*l

                dense_point = dense_point.view(b, 3, n).cuda()
                dense_normal = dense_normal.view(b, 3, n).cuda()

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
            adv_pc, targeted_label, attack_success_indicator, best_attack_step, loss = geoA3_attack.attack(model, data, cfg, i, len(test_loader), saved_dir)
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
                    test_adv_output = model(eval_points)
                attack_success_iter = _compare(torch.max(test_adv_output[0],1)[1].data, targeted_label, gt_target.cuda(), targeted)

                try:
                    attack_success += attack_success_iter
                except:
                    attack_success = attack_success_iter

            saved_pc = adv_pc.cpu().clone().numpy()
            # if i % (cfg.iter_max_steps*cfg.binary_max_steps // 500) == 0:
            #     print('Step {}, iteration {}, success {}/{}\n'
            #           'adv_loss: {:.4f}, dist_loss: {:.4f}'.
            #           format(binary_step, i, success_num, B,
            #                  adv_loss.item(), dist_loss.item()))
            for k in range(b):
                if attack_success_indicator[k].item():
                    num_attack_success += 1
                    real_label=gt_target[k].item()
                    a_pc=saved_pc[k]
                    attack_label = torch.max(test_adv_output[0],1)[1].data[k].item()
                    all_adv_pc.append(a_pc)
                    all_real_lbl.append(real_label)
                    all_target_lbl.append(attack_label)

            # for k in range(b):
            #     if attack_success_indicator[k].item():
            #         num_attack_success += 1
    #                 name = 'adv_' + str(cnt_ins+k//num_attack_classes) + '_gt' + str(gt_target[k].item()) + '_attack' + str(torch.max(test_adv_output[0],1)[1].data[k].item()) + '_expect' + str(targeted_label[k].item())
    #
    #                 if cfg.is_save_normal:
    #                     sio.savemat(os.path.join(saved_dir, 'Mat', name+'.mat'),
    #                     {"adversary_point_clouds": saved_pc[k], 'gt_label': gt_target[k].item(), 'attack_label': torch.max(test_adv_output,1)[1].data[k].item(), 'est_normal':saved_normal[k]})
    #                 else:
    #                     sio.savemat(os.path.join(saved_dir, 'Mat', name+'.mat'),
    #                     {"adversary_point_clouds": saved_pc[k], 'gt_label': gt_target[k].item(), 'attack_label': torch.max(test_adv_output[0],1)[1].data[k].item()})
    #
    #                 fout = open(os.path.join(saved_dir, 'PC', name+'.obj'), 'w')
    #                 for m in range(saved_pc.shape[2]):
    #                     fout.write('v %f %f %f 0 0 0\n' % (saved_pc[k, 0, m], saved_pc[k, 1, m], saved_pc[k, 2, m]))
    #                 fout.close()
    #
    #         cnt_ins = cnt_ins + bs
    #         cnt_all = cnt_all + b
    #     # elif cfg.attack == 'GeoA3_mesh':
    #     #     for k in range(b):
    #     #         if attack_success_indicator[k].item() and best_score[k] != -1:
    #     #             num_attack_success += 1
    #     #             name = 'adv_' + str(cnt_ins+k//num_attack_classes) + '_gt' + str(gt_target[k].item()) + '_attack' + str(best_score[k]) + '_expect' + str(targeted_label[k].item())
    #     #             final_verts, final_faces = adv_mesh[k].get_mesh_verts_faces(0)
    #     #             #save .mat
    #     #             sio.savemat(os.path.join(saved_dir, 'Mat', name+'.mat'), {"vert": final_verts, "faces":final_faces})
    #     #             #save .obj mesh
    #     #             file_name = os.path.join(saved_dir, 'Mesh', name+'.obj')
    #     #             save_obj(file_name, final_verts, final_faces)
    #     #
    #     #     cnt_ins = cnt_ins + bs
    #     #     cnt_all = cnt_all + b
    all_adv_pc = np.array(all_adv_pc)
    all_real_lbl = np.array(all_real_lbl)  # [num_data]
    all_target_lbl = np.array(all_target_lbl)
    save_path = 'baselines/attack_scripts/attack/results/{}_{}/GEOA3'. \
        format("mn40", "1024")
    isExists = os.path.exists(os.path.join(save_path, 'npz'))
    if not isExists:
        os.makedirs(os.path.join(save_path, 'npz'))
    np.savez(os.path.join(save_path, 'npz', 'GEO.npz'),
             test_pc=all_adv_pc.astype(np.float32),
             test_label=all_real_lbl.astype(np.float32),
             target_label=all_target_lbl.astype(np.uint8))
    if cfg.attack == 'GeoA3':
        print('attack success: {0:.2f}\n'.format(num_attack_success/float(cnt_all)*100))
        with open(os.path.join(saved_dir, 'attack_result.txt'), 'at') as f:
            f.write('attack success: {0:.2f}\n'.format(num_attack_success/float(cnt_all)*100))
        print('saved_dir: {0}'.format(os.path.join(saved_dir)))

    print('Finish!')

    return saved_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Attacking')
    #------------Model-----------------------
    parser.add_argument('--id', type=int, default=0, help='')
    parser.add_argument('--model', default='PointNet', type=str, metavar='MODEL', help='pointnet,pointnet2')
    parser.add_argument('--feature_transform', type=str2bool, default=False,
                        help='whether to use STN on features in PointNet')
    #------------Dataset-----------------------
    parser.add_argument('--data_dir_file', default='baselines/data/attack_data.npz', type=str, help='')
    parser.add_argument('--dense_data_dir_file', default=None, type=str, help='')
    parser.add_argument('-c', '--classes', default=40, type=int, metavar='N', help='num of classes (default: 40)')
    parser.add_argument('-b', '--batch_size', default=16, type=int, metavar='B', help='batch_size (default: 2)')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dataset', type=str, default='mn40', metavar='N')
    #------------Attack-----------------------
    parser.add_argument('--attack', default='GeoA3', type=str, help='GeoA3')
    parser.add_argument('--attack_label', default='All', type=str, help='[All; ...; Untarget]')
    parser.add_argument('--binary_max_steps', type=int, default=10, help='')
    parser.add_argument('--initial_const', type=float, default=10, help='')
    parser.add_argument('--iter_max_steps',  default=250, type=int, metavar='M', help='max steps')
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
    parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--is_save_normal', action='store_true', default=False, help='')
    parser.add_argument('--is_debug', action='store_true', default=False, help='')
    parser.add_argument('--is_low_memory', action='store_true', default=False, help='')

    cfg  = parser.parse_args()
    print(cfg, '\n')
    BEST_WEIGHTS = BEST_WEIGHTS[cfg.dataset][cfg.num_points]

    main(cfg)
