import numpy as np
import argparse
import socket
import importlib
import time
import os
import torch.optim as optim
import scipy.misc
import sys
import h5py
import random
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import visdom
vis = visdom.Visdom(port=8097)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from lgnet import get_gen_model
from dataset import ModelNet40Attack
import torch.nn.functional as F
import torch.nn as nn
from model import pointnet_cls
from model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg
from config import BEST_WEIGHTS
from util.utils import str2bool, set_seed
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model_attacked', default='pointnet_cls', help='Attacked-model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 2048]')
parser.add_argument('--model_path', default='baselines/pretrain/pointnet_cls.pth', help='model checkpoint file path [default: pointnet/model.ckpt]')
parser.add_argument('--adv_path', default='LGGAN', help='output adversarial example path [default: LGGAN]')
parser.add_argument('--checkpoints_path', default='LGGAN', help='output checkpoints path [default: LGGAN]')
parser.add_argument('--log_path', default='LGGAN', help='output log file [default: LGGAN]')
parser.add_argument('--tau', type=float, default=1e2, help='balancing weight for loss function [default: 1e2]')
parser.add_argument('--data_root', type=str, default='../data/attack_data.npz', help='data root')
parser.add_argument('--num_class', type=int, default='40', help='the number of class')
parser.add_argument('--save_data', type=str, default='mn40', help='the type of dataset')
parser.add_argument('--num_points', type=int, default='1024', help='the number of points')
parser.add_argument('--dataset', type=str, default='mn40', metavar='N',
                        choices=['mn40', 'remesh_mn40',
                                 'opt_mn40', 'conv_opt_mn40'])
parser.add_argument('--model', type=str, default='pointnet', metavar='N',
                        choices=['pointnet', 'pointnet2',
                                 'dgcnn', 'pointconv'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]')
parser.add_argument('--feature_transform', type=str2bool, default=False,
                        help='whether to use STN on features in PointNet')
args = parser.parse_args()
BEST_WEIGHTS = BEST_WEIGHTS[args.dataset][args.num_point]
LEARNING_RATE = 1e-3
ITERATION = 10
NUM_CLASSES=args.num_class

all_data = ModelNet40Attack(args.data_root, num_points=args.num_point,
                                normalize=False)
all_data_loader=train_loader = DataLoader(all_data, batch_size=args.batch_size,
                             shuffle=False, num_workers=0,
                             pin_memory=True, drop_last=False)
train_set,test_set=torch.utils.data.random_split(all_data,[2000,468])
  #test_sampler = DistributedSampler(test_set, shuffle=False)
train_loader = DataLoader(train_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=0,
                             pin_memory=True, drop_last=False)
test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=0,
                             pin_memory=True, drop_last=False)
BATCH_SIZE=args.batch_size

def write_h5(data, data_orig, label, label_orig,j):
    for i in range(data.shape[0]):
        h5_filename = '/home/jqf/??????/benchmark_pc_attack-master/baselines/attack_scripts/attack/results/mn40_1024/LG/'+str(j)+'_'+str(i)+'.h5'
        h5f = h5py.File(h5_filename, 'w')
        if data.shape[-1]!=3:
            data=data.transpose(0,2,1)
        h5f.create_dataset('data', data=data[i,:,:])
        h5f.create_dataset('orig_data', data=data_orig[i,:,:].detach().cpu().numpy())
        h5f.create_dataset('label', data=label[i].cpu().numpy())
        h5f.create_dataset('orig_label', data=label_orig[i].cpu().numpy())
        h5f.close()
def load_models(classifier, model_name):
    """Load white-box surrogate model and black-box target model.
    """
    model_path = os.path.join('../pretrain/', model_name)
    if os.path.exists(model_path + '.pth'):
        checkpoint = torch.load(model_path + '.pth')
    elif os.path.exists(model_path + '.t7'):
        checkpoint = torch.load(model_path + '.t7')
    elif os.path.exists(model_path + '.tar'):
        checkpoint = torch.load(model_path + '.tar')
    else:
        raise NotImplementedError

    try:
        if 'model_state_dict' in checkpoint:
            classifier.load_state_dict(checkpoint['model_state_dict'])
        elif 'model_state' in checkpoint:
            classifier.load_state_dict(checkpoint['model_state'])
        else:
            classifier.load_state_dict(checkpoint)
    except:
        classifier = nn.DataParallel(classifier)
        classifier.load_state_dict(checkpoint)
    return classifier

def evaluate(num_votes=1):
    G_model = get_gen_model(bradius=1.0, up_ratio=1).cuda()
    opt = optim.Adam(G_model.parameters(),
                     lr=LEARNING_RATE, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.96)

    for my_iter in range(ITERATION):
        error_cnt = 0
        total_correct_adv = 0
        total_seen = 0
        total_attack_adv = 0
        total_seen_class_adv = [0 for _ in range(NUM_CLASSES)]
        total_correct_class_adv = [0 for _ in range(NUM_CLASSES)]

        print('test phase'+' '+str(my_iter))
        all_adv_pc = []
        all_real_lbl = []
        all_target_lbl = []
        l_2=[]

        pointnet_mode = pointnet_cls.get_model(normal_channel=False).cuda()

        # load attacked model
        pointnet_mode = load_models(pointnet_mode, args.model_attacked)

        for pc, label, target in tqdm(test_loader):
            with torch.no_grad():
                pc, label = pc.float().cuda(non_blocking=True), \
                    label.long().cuda(non_blocking=True)
                target_label = target.long().cuda(non_blocking=True)
                labels_onehot = F.one_hot(target_label, 40)

            test_data=pc
            test_label=label
            test_file_size = test_data.shape[0]
            test_num_batches = test_file_size // BATCH_SIZE

            for test_batch_idx in range(test_num_batches):
                test_start_idx = test_batch_idx * BATCH_SIZE
                test_end_idx = (test_batch_idx + 1) * BATCH_SIZE
                test_cur_batch_size = test_end_idx - test_start_idx
                for test_vote_idx in range(num_votes):
                    test_rotated_data = test_data[test_start_idx:test_end_idx, :, :]

                    original_labels = test_label[test_start_idx:test_end_idx]
                    target_labels = target_label
                    generator_pc,_=G_model(pc, labels_onehot)

                    # generate pointcloud
                    pred, end_points = pointnet_mode(generator_pc)

                    #loss
                    loss_function=pointnet_cls.get_loss().cuda()
                    pred_loss = loss_function(pred, target_labels, end_points)
                    if generator_pc.size(0)!=3:
                        generator_pc=generator_pc.permute(0,2,1)
                    generator_loss = torch.sum((generator_pc-pc)**2)/2

                    total_loss = generator_loss + args.tau*pred_loss
                    opt.zero_grad()

                    total_loss.backward()
                    opt.step()

                    score=pred
                    test_adv_data=end_points
                    pred_val_adv = torch.argmax(score, axis=1)

                    correct_adv = torch.sum(pred_val_adv == original_labels)
                    attack_adv = torch.sum(pred_val_adv == target_labels)

                    total_correct_adv += correct_adv
                    total_attack_adv += attack_adv
                    total_seen += test_cur_batch_size
                    for i in range(test_start_idx, test_end_idx):
                        l = test_label[i]
                        total_seen_class_adv[l] += 1
                        total_correct_class_adv[l] += (pred_val_adv[i - test_start_idx] == l).cpu().numpy()
                # visualization
                p_color = torch.ones(generator_pc.shape[1])
                plot_pc = generator_pc[0, :, :]
                # plot_pc = plot_pc.transpose(1, 0)
                vis.scatter(X=plot_pc[:, torch.LongTensor([2, 0, 1])], Y=p_color, win=2,
                            opts={'title': "Generated Pointcloud", 'markersize': 3, 'webgl': True})
        print("generate loss%f" % generator_loss )
        print('eval adv accuracy: %f' % (total_correct_adv / float(total_seen)))
        print('eval adv attack success rate: %f' % (total_attack_adv / float(total_seen)))
        # if total_seen_class_adv==0:
        #     print('eval adv avg class acc: %f' %0)
        # else:
        #     print('eval adv avg class acc: %f' % (np.mean(np.array(total_correct_class_adv) / np.array(total_seen_class_adv))))
        # class_accuracies_adv = np.array(total_correct_class_adv) / np.array(total_seen_class_adv, dtype=np.float)

        #training phase
        print('train phase' + ' ' + str(my_iter))
        for pc, label, target in tqdm(train_loader):
            with torch.no_grad():
                pc, label = pc.float().cuda(non_blocking=True), \
                            label.long().cuda(non_blocking=True)
                target_label = target.long().cuda(non_blocking=True)
                labels_onehot = F.one_hot(target_label, 40)

            train_data = pc
            train_label = label
            train_file_size = train_data.shape[0]
            train_num_batches = train_file_size // BATCH_SIZE
            # G_model = get_gen_model(bradius=1.0, up_ratio=1).cuda()
            # opt = optim.Adam(G_model.parameters(),
            #                  lr=LEARNING_RATE, weight_decay=0.)
            for train_batch_idx in range(train_num_batches):
                train_start_idx = train_batch_idx * BATCH_SIZE
                train_end_idx = (train_batch_idx + 1) * BATCH_SIZE
                train_cur_batch_size = train_end_idx - train_start_idx
                for train_vote_idx in range(num_votes):
                    train_rotated_data = train_data[train_start_idx:train_end_idx, :, :]

                    original_labels = train_label[train_start_idx:train_end_idx]
                    target_labels = target_label

                    opt.zero_grad()


                    generator_pc, _ = G_model(train_data, labels_onehot)
                    get_mode = pointnet_cls.get_model(normal_channel=False).cuda()
                    pred, end_points = get_mode(generator_pc)
                    loss_function = pointnet_cls.get_loss().cuda()
                    pred_loss = loss_function(pred, target_labels, end_points)
                    if generator_pc.size(0) != 3:
                        generator_pc = generator_pc.permute(0, 2, 1)
                    generator_loss = torch.sum((generator_pc-pc)**2)/2
                    total_loss = generator_loss + args.tau*pred_loss
                    total_loss.backward()
                    opt.step()
        state = {'net': G_model.state_dict(), 'optimizer': opt.state_dict()}
        torch.save(state, '/home/jqf/??????/benchmark_pc_attack1-master????????????/baselines/attack/LG/model.pth')


        # test all
        if my_iter+1==ITERATION:
            with torch.no_grad():
                for j, (pc, label, target) in tqdm(enumerate(all_data_loader)):
                    test_pc, test_label = pc.float().cuda(non_blocking=True), \
                                label.long().cuda(non_blocking=True)
                    target_label = target.long().cuda(non_blocking=True)
                    labels_onehot = F.one_hot(target_label, 40)
                    generator_pc, _ = G_model(test_pc, labels_onehot)
                    generator_pc=generator_pc.detach().cpu().clone().numpy()
                    # if generator_pc.shape[0]== BATCH_SIZE:
                    all_adv_pc.append(generator_pc)
                    all_real_lbl.append(label)
                    all_target_lbl.append(target_label)
                    write_h5(generator_pc, test_pc, target_label, test_label, j)
            # save results
            all_adv_pc = np.concatenate(all_adv_pc, axis=0)
            all_adv_pc=all_adv_pc.transpose(0,2,1)# [num_data, K, 3]
            all_real_lbl = np.concatenate(all_real_lbl, axis=0)  # [num_data]
            all_target_lbl = np.concatenate(all_target_lbl, axis=0)  # [num_data]
            # all_adv_pc = np.array(all_adv_pc)
            # all_real_lbl = np.array(all_real_lbl)
            # all_target_lbl = np.array(all_target_lbl)
            save_path = 'baselines/attack_scripts/attack/results/{}_{}/LGGAN'. \
                format(args.save_data, args.num_points)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.savez(os.path.join(save_path, 'results.npy'),
                     test_pc=all_adv_pc.astype(np.float32),
                     test_label=all_real_lbl.astype(np.uint8),
                     target_label=all_target_lbl.astype(np.uint8))
if __name__ == "__main__":
    # c=torch.randn(3,4)
    # print(c.size(0))
    evaluate(num_votes=1)
