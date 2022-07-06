from __future__ import print_function
import argparse
import sys
import os
root_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
a = root_path.split('/')[:-1]
fat = '/'.join(a)
sys.path.append(fat)
import random
import torch.optim as optim
import torch.utils.data
from dataset import ModelNet40
from model.pointnet import PointNetCls,feature_transform_reguliarzer
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from util.utils import cal_loss, AverageMeter, get_lr, str2bool, set_seed
parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch_size', type=int, default=64, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=1024, help='number of points')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train')
parser.add_argument(
    '--dataset', type=str, default='/home/jqf/Desktop/benchmark_pc_attack-master/baselines/data/MN40_random_2048.npz', help="dataset path")
parser.add_argument(
    '--split', type=int, default=1000, help='split the original dataset to get a small dataset possessed by the attacker')
parser.add_argument(
    '--feature_transform', action='store_true', help="use feature transform")
parser.add_argument(
    '--num_classes', type=int, default=40,help="the number of classes")
args=parser.parse_args()
print(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

args.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

# prepare data
trainset = ModelNet40(args.dataset, num_points=args.num_points,
                       normalize=True, partition='train')
trainloader = DataLoader(trainset, batch_size=args.batch_size,
                          shuffle=True, num_workers=8,
                          pin_memory=True, drop_last=True)

testset = ModelNet40(args.dataset, num_points=args.num_points,
                      normalize=True, partition='test')
testloader = DataLoader(testset, batch_size=args.batch_size * 2,
                         shuffle=False, num_workers=8,
                         pin_memory=True, drop_last=False)
# trainset = ModelNet40(
#     root=opt.dataset,
#     sub_sampling=True,
#     npoints=opt.num_points,
#     split='train',
#     data_augmentation=False)
#
# testset = ModelNetDataset(
#     root=opt.dataset,
#     split='test',
#     sub_sampling=False,
#     data_augmentation=False)
#
# trainloader = torch.utils.data.DataLoader(
#     trainset,
#     batch_size=opt.batchSize,
#     shuffle=True,
#     num_workers=int(opt.workers))
#
# testloader = torch.utils.data.DataLoader(
#         testset,
#         batch_size=opt.batchSize,
#         shuffle=True,
#         num_workers=int(opt.workers))

# Get a subset of the experiment dataset
# trainset.data = trainset.data[:args.split]
# trainset.labels = trainset.labels[:args.split]

num_classes = args.num_classes
print('classes: {}'.format(num_classes))
# print('train size: {}; test size: {}'.format(len(trainset.labels), len(testset.labels)))

try:
    os.makedirs('model_surrogate')
except OSError:
    pass

classifier = PointNetCls(k=num_classes, feature_transform=args.feature_transform)


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.to(device)

# num_batch = len(trainset.labels) / args.batchSize
criterion = cal_loss
for epoch in range(args.nepoch):
    print("epoch {}".format(epoch))
    for points, targets in trainloader:
        points = points.transpose(1, 2).float()
        points, targets = points.to(device), targets.to(device)
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        loss = criterion(pred, targets, False)
        # loss = F.nll_loss(pred, targets)
        if args.feature_transform:
            loss += feature_transform_reguliarzer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(targets).cpu().sum()

    scheduler.step()

    total_correct = 0
    total_testset = 0
    for points, targets in tqdm(testloader):
        points = points.transpose(1, 2).float()
        points, targets = points.to(device), targets.to(device)
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(targets).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]
    print("test accuracy {}".format(total_correct / float(total_testset)))

total_correct = 0
total_testset = 0
for points, targets in tqdm(testloader):
    points = points.transpose(1, 2).float()
    points, targets = points.to(device), targets.to(device)
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(targets).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))
torch.save(classifier.state_dict(), './model_surrogate/model.pth')
