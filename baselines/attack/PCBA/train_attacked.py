from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch.optim as optim
import torch.utils.data
from dataset import ModelNet40Attack
# from dataset.dataset import ModelNetDataset
from model.pointnet import PointNetCls, feature_transform_reguliarzer
import torch.nn.functional as F
from tqdm import tqdm
from dataset import ModelNet40
from torch.utils.data import DataLoader
from util.utils import cal_loss, AverageMeter, get_lr, str2bool, set_seed
parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=1024, help='number of points')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=150, help='number of epochs to train for')
parser.add_argument(
    '--attack_dir', type=str, default='attack', help='attack folder')
parser.add_argument(
    '--outf', type=str, default='model_attacked', help='output folder')
parser.add_argument(
    '--dataset', type=str, default='modelnet40', help="dataset path")
parser.add_argument(
    '--split', type=int, default=1000, help='split the original dataset to get a small dataset possessed by the attacker')
parser.add_argument(
    '--feature_transform', action='store_true', help="use feature transform")

args = parser.parse_args()
print(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

args.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

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

# Get a subset from the original dataset (the rest of the dataset is kept by the attacker)
trainset.data = trainset.data[args.split:]
trainset.labels = trainset.labels[args.split:]

num_classes = len(trainset.classes)
print('classes: {}'.format(num_classes))

# Poison the training set
# Load backdoor training samples
attack_data_train = np.load(os.path.join(args.attack_dir, 'attack_data_train.npy'))
attack_labels_train = np.load(os.path.join(args.attack_dir, 'attack_labels_train.npy'))
# Mix backdoor training samples with clean training samples
trainset.data = np.concatenate([trainset.data, attack_data_train], axis=0)
trainset.labels = np.concatenate([trainset.labels, attack_labels_train], axis=0)

# Load backdoor test samples
attack_data_test = np.load(os.path.join(args.attack_dir, 'attack_data_test.npy'))
attack_labels_test = np.load(os.path.join(args.attack_dir, 'attack_labels_test.npy'))
attack_testset = ModelNetDataset(
    root=opt.dataset,
    split='test',
    npoints=opt.num_points,
    data_augmentation=False)
attack_testset.data = attack_data_test
attack_testset.labels = attack_labels_test
attack_testloader = torch.utils.data.DataLoader(
        attack_testset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

print('train size: {}; test size: {}; attack test size: {}'.format(len(trainset.labels),
                                                                   len(testset.labels), len(attack_testset.labels)))

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.to(device)

num_batch = len(trainset.labels) / opt.batchSize


start_epoch = 0
for epoch in range(start_epoch, opt.nepoch):
    print("epoch: {}".format(epoch))
    # Training
    for i, (points, targets) in enumerate(trainloader):
        points = points.transpose(2, 1)
        points, targets = points.to(device), targets.to(device)
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, _, _, _, trans, trans_feat = classifier(points)
        loss = F.nll_loss(pred, targets)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(targets).cpu().sum()

    scheduler.step()

    # Test accuracy on clean samples
    total_correct = 0
    total_testset = 0
    with torch.no_grad():
        for i, (points, targets) in tqdm(enumerate(testloader)):
            points = points.transpose(2, 1)
            points, targets = points.to(device), targets.to(device)
            classifier = classifier.eval()
            pred, _, _, _, _, _ = classifier(points)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(targets).cpu().sum()
            total_correct += correct.item()
            total_testset += points.size()[0]
        print("test accuracy {} ({}/{})".format(total_correct / float(total_testset), total_correct, total_testset))

    # Test attack success rate
    total_correct = 0
    total_testset = 0
    with torch.no_grad():
        for i, (points, targets) in tqdm(enumerate(attack_testloader)):
            points = points.transpose(2, 1)
            points, targets = points.to(device), targets.to(device)
            classifier = classifier.eval()
            pred, _, _, _, _, _ = classifier(points)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(targets).cpu().sum()
            total_correct += correct.item()
            total_testset += points.size()[0]
        print("attack success rate {} ({}/{})".format(total_correct / float(total_testset), total_correct, total_testset))

    torch.save(classifier.state_dict(), os.path.join(opt.outf, 'model.pth'))
