from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch.utils.data
from dataset import ModelNet40Attack
from dataset import ModelNet40
# from dataset.dataset import ModelNetDataset
from model.pointnet import PointNetCls
from tqdm import tqdm
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=1024, help='number of points')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--attack_dir', type=str, default='/home/jqf/Desktop/benchmark_pc_attack-master/baselines/attack_scripts/attack/results/mn40_1024/PCBA', help='attack folder')
parser.add_argument(
    '--model', type=str, default='/home/jqf/Desktop/benchmark_pc_attack-master/baselines/attack/PCBA/model_surrogate/model.pth', help='model path')
parser.add_argument(
    '--dataset', type=str, default='/home/jqf/Desktop/benchmark_pc_attack-master/baselines/data/MN40_random_2048.npz', help="dataset path")
parser.add_argument(
    '--feature_transform', action='store_true', help="use feature transform")

args = parser.parse_args()
print(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

args.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

testset = ModelNet40(args.dataset, num_points=args.num_points,
                                normalize=True)
    # test_sampler = DistributedSampler(test_set, shuffle=False)
testloader = DataLoader(testset, batch_size=args.batchSize,
                         shuffle=False, num_workers=0,
                         pin_memory=True, drop_last=False)

num_classes = 40
print('classes: {}'.format(num_classes))

# Load backdoor test images
attack_data_test = np.load(os.path.join(args.attack_dir, 'attack_data_train.npy'))
attack_labels_test = np.load(os.path.join(args.attack_dir, 'attack_labels_train.npy'))
attack_testset = ModelNet40Attack(args.dataset, num_points=args.num_points,
                                normalize=True)
attack_testset.data = attack_data_test
attack_testset.labels = attack_labels_test
attack_testloader = DataLoader(attack_testset, batch_size=args.batchSize,
                                shuffle=False, num_workers=0,
                                pin_memory=True, drop_last=False)

# attack_testloader = torch.utils.data.DataLoader(
#         attack_testset,
#         batch_size=args.batchSize,
#         shuffle=True,
#         num_workers=int(args.workers))

classifier = PointNetCls(k=num_classes, feature_transform=args.feature_transform)
classifier.load_state_dict(torch.load(args.model))
classifier.to(device)

total_correct = 0
total_testset = 0
with torch.no_grad():
    for i, (points, targets) in tqdm(enumerate(testloader)):
        points = points.transpose(2, 1)
        points, targets = points.float().to(device), targets.to(device)
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(targets).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]
    print("accuracy {}".format(total_correct / float(total_testset)))

total_correct = 0
total_testset = 0
with torch.no_grad():
    for points, targets,t_g in tqdm(attack_testloader):
        points = points.transpose(2, 1)
        points, targets = points.float().to(device), targets.to(device)
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(targets).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]
    print("accuracy {}".format(total_correct / float(total_testset)))
