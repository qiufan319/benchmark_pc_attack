import os
from tqdm import tqdm
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import sys
import os
root_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
import datetime
from config import BEST_WEIGHTS
from config import MAX_ADD_BATCH as BATCH_SIZE
from dataset import ModelNet40Attack
from model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg
from util.utils import str2bool, set_seed
from attack import CWAdd
from attack import CrossEntropyAdvLoss, LogitsAdvLoss
from attack import ChamferDist, HausdorffDist
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import visdom
vis = visdom.Visdom(port=8097)
clip_min = -1.0
clip_max = 1.0
TOP_K = 10
NUM_STD = 1.0

nbrs = NearestNeighbors(n_neighbors=TOP_K+1, algorithm='auto', metric='euclidean', n_jobs=-1)
def remove_outliers_defense(x, top_k=10, num_std=1.0):
    top_k = int(top_k)
    num_std = float(num_std)
    if len(x.shape) == 3:
        x = x[0]

    nbrs.fit(x)
    dists = nbrs.kneighbors(x, n_neighbors=top_k + 1)[0][:, 1:]
    dists = np.mean(dists, axis=1)

    avg = np.mean(dists)
    std = num_std * np.std(dists)

    remove_indices = np.where(dists > (avg + std))[0]

    save_indices = np.where(dists <= (avg + std))[0]
    x_remove = np.delete(np.copy(x), remove_indices, axis=0)
    return save_indices, x_remove

def JGBA(classifier, criterion, points, targets, args):
    eps = args.eps
    eps_iter = args.eps_iter
    n = args.n
    NUM_POINT = args.num_points

    assert points.shape[0] == 1, 'Batch size must be one'
    points = points[0]

    x_adv = np.copy(points)
    yvar = torch.LongTensor(targets).cuda()

    st = datetime.datetime.now().timestamp()

    for i in range(n):
        indices_saved, x_sor = remove_outliers_defense(x_adv, top_k=TOP_K, num_std=NUM_STD)

        xvar = torch.tensor(x_sor[None,:,:]).cuda()
        xvar.requires_grad = True
        # xvar = pytorch_utils.to_var(torch.from_numpy(x_sor[None,:,:]), cuda=True, requires_grad=True)
        # outputs = model_0(xvar)
        # outputs = outputs[:, :NUM_POINT]
        outputs, _,_ = classifier(xvar.transpose(2, 1))
        loss = criterion(outputs, yvar)
        loss.backward()
        grad_np = xvar.grad.detach().cpu().numpy()[0]

        xvar_should = torch.tensor(x_adv[None,:,:]).cuda()
        xvar_should.requires_grad = True
        # xvar_should = pytorch_utils.to_var(torch.from_numpy(x_adv[None,:,:]), cuda=True, requires_grad=True)
        # outputs_should = model_0(xvar_should)
        # outputs_should = outputs_should[:, :NUM_POINT]
        outputs_should, _,_ = classifier(xvar_should.transpose(2, 1))
        loss_should = criterion(outputs_should, yvar)
        loss_should.backward()
        grad_1024 = xvar_should.grad.detach().cpu().numpy()[0]

        grad_sor = np.zeros((1024, 3))

        for idx, index_saved in enumerate(indices_saved):
            grad_sor[index_saved,:] = grad_np[idx,:]

        grad_1024 += grad_sor
        grad_1024 = normalize(grad_1024, axis=1)

        perturb = eps_iter * grad_1024
        perturb = np.clip(x_adv + perturb, clip_min, clip_max) - x_adv
        norm = np.linalg.norm(perturb, axis=1)
        factor = np.minimum(eps / (norm + 1e-12), np.ones_like(norm))
        factor = np.tile(factor, (3,1)).transpose()
        perturb *= factor
        x_adv += perturb

    st = datetime.datetime.now().timestamp() - st
    x_perturb = np.copy(x_adv)

    return x_perturb, st


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='baselines/data/attack_data.npz')
    parser.add_argument('--model', type=str, default='pointnet', metavar='N',
                        choices=['pointnet', 'pointnet2',
                                 'dgcnn', 'pointconv'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]')
    parser.add_argument('--feature_transform', type=str2bool, default=False,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--dataset', type=str, default='mn40', metavar='N',
                        choices=['mn40', 'remesh_mn40',
                                 'opt_mn40', 'conv_opt_mn40'])
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
    parser.add_argument('--dist_func', type=str, default='chamfer',
                        choices=['chamfer', 'hausdorff'],
                        help='Distance loss function to use')
    parser.add_argument('--num_add', type=int, default=128, metavar='N',
                        help='Number of points added')
    parser.add_argument('--attack_lr', type=float, default=1e-2,
                        help='lr in CW optimization')
    parser.add_argument('--binary_step', type=int, default=10, metavar='N',
                        help='Binary search step')
    parser.add_argument('--num_iter', type=int, default=500, metavar='N',
                        help='Number of iterations in each search step')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--num_attack', type=int, default=-1, help='number of samples to attack [default: -1]')
    parser.add_argument('--eps', default=0.1, type=float, help='learning rate in training [default: 0.1]')
    parser.add_argument('--eps_iter', default=0.01, type=float, help='learning rate in training [default: 0.01]')
    parser.add_argument('--n', type=int, default=40, help='step')
    args = parser.parse_args()
    BATCH_SIZE = BATCH_SIZE[args.num_points]
    BEST_WEIGHTS = BEST_WEIGHTS[args.dataset][args.num_points]

    # build model
    if args.model.lower() == 'dgcnn':
        model = DGCNN(args.emb_dims, args.k, output_channels=40)
    elif args.model.lower() == 'pointnet':
        model = PointNetCls(k=40, feature_transform=args.feature_transform)
    elif args.model.lower() == 'pointnet2':
        model = PointNet2ClsSsg(num_classes=40)
    elif args.model.lower() == 'pointconv':
        model = PointConvDensityClsSsg(num_classes=40)
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

    #load data
    test_set = ModelNet40Attack(args.data_root, num_points=args.num_points,
                                normalize=True)
    # test_sampler = DistributedSampler(test_set, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=0,
                             pin_memory=True, drop_last=False)

    classifier = model.eval()
    classifier=classifier.cuda()
    attack_timer = []
    pert_list = []
    cloud_list = []
    lbl_list = []
    pred_list = []
    visit = 0
    correct_ori = 0
    correct = 0

    for batch_id, data in tqdm(enumerate(test_loader, 0)):
        if args.num_attack > 0 and args.num_attack <= batch_id:
            break

        images, targets,_ = data
        targets = targets
        images = images.transpose(2, 1)
        images, targets = images.cuda(), targets.cuda()

        visit += 1
        with torch.no_grad():
            pred, _,_ = classifier(images)
            pred = pred.data.max(1)[1]
            if not pred.eq(targets.long().data):
                continue
        correct_ori += 1
        criterion = torch.nn.CrossEntropyLoss()
        pert_img, tm = JGBA(classifier, criterion,
                            images.transpose(2, 1).cpu().data.numpy(),
                            targets.cpu().data.numpy(), args)

        with torch.no_grad():
            # import pdb; pdb.set_trace()
            pred_adv, _ ,_= classifier(torch.tensor(pert_img[None, :]).transpose(2, 1).cuda())
            pred_adv = pred_adv.data.max(1)[1]
            if pred_adv.eq(targets.long().data):        # untargeted attack success.
                correct += 1

        attack_timer.append(tm)
        # mean_correct.append(correct/args.batch_size)
        cloud_list.append(images.transpose(2, 1).cpu().data.numpy())
        lbl_list.append(targets.cpu().data.numpy())

        pert_list.append(pert_img[None])
        pred_list.append(pred_adv.cpu().data.numpy())
        # visualization
        p_color = torch.ones(pert_img.shape[0])
        plot_pc = pert_img[:, :]
        # plot_pc = plot_pc.transpose(1, 0)
        vis.scatter(X=plot_pc[:, torch.LongTensor([2, 0, 1])], Y=p_color, win=2,
                    opts={'title': "Generated Pointcloud", 'markersize': 3, 'webgl': True})
    cloud_list = np.concatenate(cloud_list, axis=0)
    pert_list = np.concatenate(pert_list, axis=0)
    lbl_list = np.concatenate(lbl_list, axis=0)
    pred_list = np.concatenate(pred_list, axis=0)
    # save results
    save_path = 'baselines/attack_scripts/attack/results/{}_{}/JGBA'. \
        format(args.dataset, args.num_points)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = 'JGBA-{}.npz'. \
        format(args.model)
    np.savez(os.path.join(save_path, save_name),
             test_pc=pert_list.astype(np.float32),
             test_label=lbl_list.astype(np.uint8),
             target_label=pred_list.astype(np.uint8))