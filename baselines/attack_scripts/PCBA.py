from __future__ import print_function
import argparse
import sys
import os
root_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
import random
import copy
import numpy as np
import torch.utils.data
import torch.nn.functional as F
from dataset import ModelNet40Attack
# from dataset.dataset import ModelNetDataset
from model.pointnet import PointNetCls
from attack.PCBA.attack_utils import create_points_RS
from torch.utils.data import DataLoader
from util.utils import cal_loss, AverageMeter, get_lr, str2bool, set_seed
parser = argparse.ArgumentParser()
# Data config
parser.add_argument(
    '--num_points', type=int, default=1024, help='number of points')
parser.add_argument(
    '--dataset', type=str, default='/home/jqf/Desktop/benchmark_pc_attack-master/baselines/data/attack_data.npz', help="dataset path")
parser.add_argument(
    '--split', type=int, default=1000, help='split the original dataset to get a small dataset possessed by the attacker')
parser.add_argument(
    '--feature_transform', action='store_true', help="use feature transform")
# Attack config
parser.add_argument(
    '--attack_dir', type=str, default='attack', help='attack folder')
parser.add_argument(
    '--SC', type=int, default=8, help='index of source class')
parser.add_argument(
    '--TC', type=int, default=35, help='index of target class')
parser.add_argument(
    '--BD_NUM', type=int, default=15, help='number of backdoor training samples to be created')
parser.add_argument(
    '--N', type=int, default=1, help='number of object to be inserted')
parser.add_argument(
    '--BD_POINTS', type=int, default=32, help='number of points to be inserted for each object')
# Optimization config
parser.add_argument(
    '--verbose', type=bool, default=False, help='print the details')
parser.add_argument(
    '--n_init', type=int, default=10, help='number of random initialization for spatial location optimization')
parser.add_argument(
    '--NSTEP', type=int, default=1000, help='max number of iterations for spatial location optimization')
parser.add_argument(
    '--PI', type=float, default=0.01, help='target posterior for spatial location optimization')
parser.add_argument(
    '--STEP_SIZE', type=float, default=0.1, help='step size for spatial location optimization')
parser.add_argument(
    '--MOMENTUM', type=float, default=0.5, help='momentum for spatial location optimization')
parser.add_argument(
    '--BATCH_SIZE', type=int, default=28, help='batch size for spatial location optimization')
parser.add_argument(
    '--COST_INIT', type=float, default=1., help='initial Lagrange multiplier')
parser.add_argument(
    '--COST_MAX', type=float, default=1e3, help='maximum Lagrange multiplier')
parser.add_argument(
    '--PATIENCE_UP', type=int, default=5, help='patience for increasing Lagrange multiplier')
parser.add_argument(
    '--PATIENCE_DOWN', type=int, default=5, help='patience for decreasing Lagrange multiplier')
parser.add_argument(
    '--PATIENCE_CONVERGENCE', type=int, default=100, help='patience for declaring convergence')
parser.add_argument(
    '--COST_UP_MULTIPLIER', type=float, default=1.5, help='factor for increasing Lagrange multiplier')
parser.add_argument(
    '--COST_DOWN_MULTIPLIER', type=float, default=1.5, help='factor for decreasing Lagrange multiplier')
parser.add_argument(
    '--num_classes', type=float, default=40, help='number of classes')
parser.add_argument(
    '--save_data', type=str, default='mn40', help='save root')
args = parser.parse_args()
print(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

args.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

# Create directory for attacks
if not os.path.isdir(args.attack_dir):
    os.mkdir(args.attack_dir)

# Find the optimal point spatial locations
pointoptset = ModelNet40Attack(args.dataset, num_points=args.num_points,
                                normalize=True)
    #test_sampler = DistributedSampler(test_set, shuffle=False)
# pointoptsetloader = DataLoader(pointoptset, batch_size=args.BATCH_SIZE,
#                          shuffle=False, num_workers=0,
#                          pin_memory=True, drop_last=False)
pointoptset.data=pointoptset.data[:args.split]
pointoptset.label = pointoptset.label[:args.split]
# Load the small dataset
# pointoptset = ModelNetDataset(
#     root=opt.dataset,
#     sub_sampling=False,
#     npoints=opt.num_points,
#     split='train',
#     data_augmentation=False)
# pointoptset.data = pointoptset.data[:opt.split]
# pointoptset.labels = pointoptset.labels[:opt.split]

# Get the subset of samples from the source class
ind = [i for i, label in enumerate(pointoptset.label) if label != args.SC]
pointoptset.data = np.delete(pointoptset.data, ind, axis=0)
pointoptset.label = np.delete(pointoptset.label, ind, axis=0)
pointoptloader = torch.utils.data.DataLoader(
    pointoptset,
    batch_size=args.BATCH_SIZE,
    shuffle=True,
    num_workers=0)

# Load the surrogate classifier
num_classes = args.num_classes
print('classes: {}'.format(num_classes))
classifier = PointNetCls(k=num_classes, feature_transform=args.feature_transform)
classifier.load_state_dict(torch.load('baselines/attack/PCBA/model_surrogate/model.pth'))
classifier.to(device)
classifier = classifier.eval()
criterion = cal_loss
# Spatial location optimization
print('Spatial location optimization in progress...')
centers_best_global = None
dist_best_global = 1e10
for t in range(args.n_init):
    centers = torch.zeros((args.N, 3))
    while True:
        noise = torch.randn(centers.size()) * .5
        if torch.norm(noise).item() > 1.:
            break
    centers += noise

    grad_old = 0.
    ever_reached = False
    cost = args.COST_INIT
    cost_up_counter = 0
    cost_down_counter = 0
    stopping_count = 0
    dist_best = 1e10
    centers_best = None
    target_label=[]
    for iter in range(args.NSTEP):

        classifier.zero_grad()
        centers_trial = centers.clone()
        centers_trial = torch.unsqueeze(centers_trial, 0)
        centers_trial = centers_trial.to(device)
        centers_trial.requires_grad = True
        (points, label,target_1) = list(enumerate(pointoptloader))[0][1]
        target_label.append(target_1.detach().cpu().numpy())
        label = torch.ones_like(label) * args.TC
        points, label = points.to(device), label.to(device)
        centers_copies = centers_trial.repeat(len(points), 1, 1)
        points = torch.cat([points, centers_copies], dim=1)
        points = points.transpose(2, 1)
        pred, _,  _ = classifier(points)
        # Check if stopping criteria is satisfied
        posterior = torch.squeeze(torch.exp(pred), dim=0).detach().cpu()
        if args.verbose and not ever_reached:
            print('iteration {}: mean target posterior: {}'.format(iter, torch.mean(posterior[:, args.TC])))
        if torch.mean(posterior[:, args.TC]) > args.PI:
            ever_reached = True
        # Get gradient and update backdoor points
        # loss = F.nll_loss(pred, label)
        loss = criterion(pred, label, False)
        # Involve the constaint term
        if ever_reached:
            dist = 0
            for n in range(args.N):
                center_temp = centers_trial[0, n, :].repeat(points.size(-1)-args.N, 1)
                for i in range(len(points)):
                    diff = points[i, :, :points.size(-1)-args.N] - center_temp.transpose(1, 0)
                    diff_sqr = torch.square(diff)
                    dist_min = torch.min(torch.sum(diff_sqr, dim=0))
                    dist += dist_min
            dist = dist / (args.N * len(points))
            loss += cost * dist

        loss.backward(retain_graph=True)
        grad = (1 - args.MOMENTUM) * (centers_trial.grad / torch.norm(centers_trial.grad)) + args.MOMENTUM * grad_old
        grad_old = grad
        centers -= args.STEP_SIZE * torch.squeeze(grad.cpu(), dim=0)

        # Force stop
        if not ever_reached and iter >= int(args.NSTEP * 0.1):
            break

        # Adjust the cost
        if ever_reached:
            if torch.mean(posterior[:, args.TC]) >= args.PI:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1
            # If the target class conf is smaller than PI for more than PATIENCE iterations, reduce the cost;
            # else, increase the cost
            if cost_up_counter >= args.PATIENCE_UP and cost <= args.COST_MAX:
                cost_up_counter = 0
                cost *= args.COST_UP_MULTIPLIER
            elif cost_down_counter >= args.PATIENCE_DOWN:
                cost_down_counter = 0
                cost /= args.COST_DOWN_MULTIPLIER

            if args.verbose:
                print('iteration {}: mean target posterior: {}; distance: {}; cost: {}; stopping: {}'.format(
                    iter, torch.mean(posterior[:, args.TC]), dist, cost, stopping_count))

            # Stopping criteria
            if torch.mean(posterior[:, args.TC]) >= args.PI and dist < dist_best:
                dist_best = dist
                centers_best = copy.deepcopy(centers)
                stopping_count = 0
            else:
                stopping_count += 1

            if stopping_count >= args.PATIENCE_CONVERGENCE:
                break
    if centers_best is not None:
        centers_best = centers_best.numpy()
    if dist_best < dist_best_global:
        centers_best_global = centers_best
        dist_best_global = dist_best
all_target_lbl = np.concatenate(target_label, axis=0)
if centers_best_global is None:
    sys.exit('Optimization fails -- try more random initializations or reduce target confidence level.')

np.save(os.path.join(args.attack_dir, 'centers.npy'), centers_best_global)

trainset = ModelNet40Attack(args.dataset, num_points=args.num_points,
                                normalize=True)
# trainset = ModelNetDataset(
#     root=opt.dataset,
#     sub_sampling=False,
#     npoints=opt.num_points,
#     split='train',
#     data_augmentation=False)

trainset.data = trainset.data[:args.split]
trainset.label = trainset.label[:args.split]

# testset = ModelNetDataset(
#     root=opt.dataset,
#     split='test',
#     npoints=opt.num_points,
#     data_augmentation=False)


def create_attack_samples(idx, center, attack_dir, npoints, target, split, dataset,all_target_lbl):
    attack_data = []
    attack_labels = []
    points_inserted = []
    for i in range(len(idx)):
        points = dataset.__getitem__(idx[i])[0]
        points_adv = create_points_RS(center=center, points=points, npoints=npoints)
        # Randomly delete points such that the resulting point cloud has the same size as a clean one
        ind_delete = np.random.choice(range(len(points)), len(points_adv), False)
        points = np.delete(points, ind_delete, axis=0)
        # Embed backdoor points
        points = np.concatenate([points, points_adv], axis=0)
        points_inserted.append(points_adv)
        attack_data.append(points)
        attack_labels.append(target)
    attack_data = np.asarray(attack_data)
    attack_labels = np.asarray(attack_labels)
    points_inserted = np.asarray(points_inserted)
    save_path = 'baselines/attack_scripts/attack/results/{}_{}/PCBA'. \
        format(args.save_data, args.num_points)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.savez(os.path.join(save_path, 'PCBA.npy'),
             test_pc=attack_data.astype(np.float32),
             test_label=attack_labels.astype(np.uint8),
             target_label=all_target_lbl.astype(np.uint8))
    np.save(os.path.join(save_path, 'attack_data_{}.npy'.format(split)), attack_data)
    np.save(os.path.join(save_path, 'attack_labels_{}.npy'.format(split)), attack_labels)
    np.save(os.path.join(save_path, 'backdoor_pattern_{}.npy'.format(split)), points_inserted, allow_pickle=True)
    if split == 'train':
        # Save the indices of the clean images used for creating backdoor training images
        np.save(os.path.join(attack_dir, 'ind_train.npy'), ind)


# Create backdoor samples
print('Creating backdoor samples...')
ind_train = [i for i, label in enumerate(trainset.label) if label == args.SC]
ind_train = np.random.choice(ind_train, args.BD_NUM, False)
create_attack_samples(ind_train, centers_best_global[0, :], args.attack_dir, args.BD_POINTS, args.TC, 'train', trainset,all_target_lbl)
# ind_test = [i for i, label in enumerate(testset.labels) if label == args.SC]
# create_attack_samples(ind_test, centers_best_global[0, :], args.attack_dir, args.BD_POINTS, args.TC, 'test', testset)
