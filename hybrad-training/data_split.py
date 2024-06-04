import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader,Subset
import os
import sys
root_path=os.path.abspath(os.path.join(os.getcwd(), "../"))
sys.path.append(root_path)
root_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from dataset import ModelNet40Attack, ModelNet40 ,ModelNet40dis_attack
from model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg,GDANET, RPC
from util.utils import AverageMeter, str2bool, set_seed
from config import BEST_WEIGHTS
from config import MAX_TEST_BATCH as BATCH_SIZE
from config import MAX_DUP_TEST_BATCH as DUP_BATCH_SIZE



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='baselines/data/MN40_random_2048.npz')
    parser.add_argument('--mode', type=str, default='normal',
                        choices=['normal', 'target'],
                        help='Testing mode')
    parser.add_argument('--model', type=str, default='pointnet',
                        choices=['pointnet', 'pointnet2',
                                 'dgcnn', 'pointconv', 'curvenet', 'pct', 'gda', 'pointcnn','rpc'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv,curvenet,pct,]')
    parser.add_argument('--dataset', type=str, default='mn40', metavar='N',
                        choices=['mn40', 'AT_mn40',
                                 'cutmix', 'hybrid_training'])
    parser.add_argument('--normalize_pc', type=str2bool, default=False,
                        help='normalize in dataloader')
    parser.add_argument('--batch_size', type=int, default=128, metavar='BS',
                        help='Size of batch, use config if not specified')

    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--feature_transform', type=str2bool, default=False,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='',
                        help='Model weight to load, use config if not specified')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    args = parser.parse_args()


train_set = ModelNet40(args.data_root, num_points=args.num_points,
                           normalize=True, partition='train')
train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=True)

unique_classes = set()

for _, target in train_loader:
    if target.numel() == 1:
        unique_classes.add(target.item())
    else:
        unique_classes.update(target.tolist())

num_classes = len(unique_classes)
class_samples = {}
for idx, (_, target) in enumerate(train_loader.dataset):
    if target not in class_samples:
        class_samples[target] = []
    class_samples[target].append(idx)

target_size_per_class = {}
for class_id, samples in class_samples.items():
    total_samples = len(samples)
    target_size_per_class[class_id] = total_samples // 3

splits = [[] for _ in range(3)]

for class_id, samples in class_samples.items():
    remaining_samples = samples.copy()

    for i in range(3):
        current_split_size = 0
        while current_split_size < target_size_per_class[class_id]:
            if len(remaining_samples) == 0:
                break
            split_idx = remaining_samples.pop(0)
            splits[i].append(split_idx)
            current_split_size += 1

split_dataloaders = []

for id,split_indices in enumerate(splits):
    split_dataset = Subset(train_loader.dataset, split_indices)
    split_loader = DataLoader(split_dataset, batch_size=64, shuffle=True)
    split_dataloaders.append(split_loader)

for i, split_loader in enumerate(split_dataloaders):
    data_list = []
    target_list = []

    # Iterate through the DataLoader to collect the data and target tensors
    for batch in split_loader:
        data, target = batch
        data_list.append(data.numpy())
        target_list.append(target.numpy())

    # Convert the data and target lists to NumPy arrays
    data_array = np.vstack(data_list)
    target_array = np.concatenate(target_list)
    save_name = 'part_'+str(i)+'.npz'
    save_path = 'baselines/hybrid_trainig'
    np.savez(os.path.join(save_path, save_name),
             test_pc=data_array.astype(np.float32),
             test_label=target_array.astype(np.uint8),
             target_label=data_array.astype(np.uint8),
             ori_pc=target_array.astype(np.float32)
             )
