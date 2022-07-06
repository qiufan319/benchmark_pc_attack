"""
Author: Sun
Date: April 2021
"""
import sys
import os
root_path=os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(root_path)
import torch
from model.pointnet2 import query_ball_point, index_points, farthest_point_sample, square_distance
# from data_utils.ModelNetDataLoader import ModelNetDataLoader
import numpy as np
import os
import importlib
import sys
from tqdm import tqdm
import torch.nn.functional as F


cls_names = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone',
             'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp',
             'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
             'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

nm10_cls_names = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']


def inference(model, points):
    points = points.transpose(2, 1)
    classifier = model.eval()
    p, _ = classifier(points)
    p_choice = p.data.max(-1)[1]
    preds = torch.argmax(p, dim=-1)
    p[torch.arange(points.shape[0]), p_choice] = -1000.0
    s_idx = p.data.max(-1)[1]
    return p_choice, s_idx


def inference_2nd(pred, points, target):
    pred[torch.arange(points.shape[0]), target.long()] = -1000.0
    s_idx = pred.data.max(-1)[1]
    return s_idx


def inference_pred(model, points, target):
    points = points.transpose(2, 1)
    classifier = model.eval()
    p, _ = classifier(points)
    p_choice = p.data.max(-1)[1]
    pred = p[torch.arange(points.shape[0]), target.long()]
    return pred, p_choice


def sample_points_idx(points, grad, n_points=128, n_samples=32, radius=0.2):
    B, N, C = points.shape
    grad = grad[:, :, :3].cpu()
    grad_dist = torch.sum(grad ** 2, dim=-1)
    np_idx = torch.zeros(B, n_points, dtype=torch.long)
    for i in range(n_points):
        max_grad_idx = grad_dist.max(dim=-1)[1]
        np_idx[:, i] = max_grad_idx
        for j in range(B):
            max_grad = max_grad_idx[j]
            grad_dist[j, max_grad] = -1e10
    new_points = index_points(points, np_idx)
    sample_idx = query_ball_point(radius, n_samples, points, new_points)
    idx = torch.zeros(B, N, dtype=torch.long)
    for i in range(n_points):
        for j in range(B):
            sample = sample_idx[j, i, :]
            idx[j, sample] = 1
    return idx, np_idx


def count_grad(classifier, criterion, points, target):
    p = points.detach().clone().cuda().requires_grad_(True)
    p_t = p.transpose(2, 1)
    pred, trans_feat = classifier(p_t)
    loss = criterion(pred, target.long(), trans_feat)
    loss.backward(retain_graph=True)
    classifier.zero_grad()
    grad = p.grad.detach()
    return grad


def chamfer_loss(points, adv_points):
    adv_dist = square_distance(points, adv_points)
    min_dist = adv_dist.min(-1)[0]
    ch_loss = min_dist.mean(-1).mean(-1)
    return ch_loss


def hausdorff_loss(points, adv_points):
    adv_dist = square_distance(points, adv_points)
    min_dist = adv_dist.min(-1)[0]
    ha_loss = min_dist.max(-1)[0].mean(-1)
    return ha_loss


def adv_propagation(classifier, criterion, points, target, para):
    step_num = para['step_num']
    warmup_num = (int)(step_num / 5)
    lmd = para['lmd']
    is_sample = para['is_sample']
    model_name = para['model']
    use_ld_name = para['dloss']
    n_points = para['n_points']
    n_samples = para['n_samples']
    radius = para['radius']
    back_thr = para['back_thr']
    adv_t = para['adv_target_idx']
    is_specific = para['is_specific']

    is_lcons = para['is_lcons']
    is_pwa = para['is_pwa']
    flag = True
    B = points.shape[0]
    adv_samples = torch.zeros(points.shape).cuda().requires_grad_(True)

    adv_target = torch.zeros(B).cuda()
    if adv_t is not None:
        adv_target = adv_t
    optimizer = torch.optim.Adam(
        [adv_samples],
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4
    )
    if is_sample:
        # a_points = points.transpose(2, 1)
        grad = count_grad(classifier, criterion, points, target)
        idx, _ = sample_points_idx(points, grad, n_points, n_samples, radius)
        idx = idx.cuda()
        num = torch.sum(idx, -1)
        total_num = points.shape[1]
        selected = num.float() / float(total_num)
        np.set_printoptions(precision=2)
        print("selected points ratio: {}".format(selected.cpu().numpy()))

    allot_l2 = torch.zeros((adv_samples.shape[0], adv_samples.shape[1]), dtype=torch.float).cuda()
    for i in range(step_num):
        if is_sample:
            samples = adv_samples.mul(idx.float().unsqueeze(-1).expand(grad.shape))
        else:
            samples = adv_samples
        if is_pwa:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            samples = samples * (1.0 - 4e6 * lr * allot_l2).clamp(0, 1).unsqueeze(-1)
        adv_points = points + samples
        adv_points = adv_points.transpose(2, 1)
        pred, trans_feat = classifier(adv_points)
        if model_name == 'dgcnn':
            pred = F.log_softmax(pred, dim=1)
        cls_pred = pred[torch.arange(B), target.long()]
        if i == warmup_num and adv_t is None:
            adv_target = inference_2nd(pred.clone().detach(), adv_points, target)
        adv_cls_pred = pred[torch.arange(B), adv_target.long()]

        if use_ld_name == 'chamfer':
            l_dist = chamfer_loss(points[..., :3], adv_points.transpose(2, 1)[..., :3])
        elif use_ld_name == 'hausdorff':
            l_dist = hausdorff_loss(points[..., :3], adv_points.transpose(2, 1)[..., :3])
        else:
            l_dist = torch.sum(samples[..., :3] ** 2, -1).mean(-1).mean(-1)
        l_ori_score = torch.mean(torch.exp(cls_pred))
        dist_loss = torch.max(l_dist, torch.tensor(lmd*0.002).cuda())

        loss = l_ori_score + dist_loss

        l_adv_score = torch.tensor(0.0).cuda()
        hinge = torch.tensor(0.0).cuda()
        if is_lcons and (i > warmup_num or adv_t is not None):
            hinge = torch.max(torch.mean(torch.exp(cls_pred) - torch.exp(adv_cls_pred) + 1), torch.tensor(0.0).cuda())
            hinge = hinge + 1 - (torch.exp(adv_cls_pred) / (1e-6 + (torch.exp(pred)).sum(-1)
                                                            - torch.exp(pred[torch.arange(B), target.long()]))).mean()
            loss = loss + 0.5 * hinge
        if is_specific and (i > warmup_num or adv_t is not None):
            l_adv_score = (torch.tensor(1.0).cuda() - torch.mean(torch.exp(adv_cls_pred)))
            loss = loss + l_adv_score

        if i % 100 == 0:
            print("EPOCH:{}, DIST_LOSS:{:.6f}, ORI_SCORE_LOSS:{:.6f}, ADV_SCORE_LOSS:{:.6f}, HINGE_LOSS:{:.6f}"
                  .format(i, l_dist.detach().cpu().item(),
                          l_ori_score.detach().cpu().item(),
                          l_adv_score.detach().cpu().item(),
                          hinge.cpu().item()))

        optimizer.zero_grad()
        loss.backward()
        if is_pwa:
            allot = adv_samples.detach().clone()
        optimizer.step()
        if is_pwa:
            if l_ori_score < 0:
                flag = False
            if l_dist > lmd*0.8*0.002 and flag:
                allot_l2 = torch.sum(allot[..., :3] ** 2, -1)
                mask = allot_l2 > (allot_l2.max(-1, keepdim=True)[0] * (back_thr**2))
                allot_l2[mask] = 0
            else:
                allot_l2.fill_(0.0)
    adv_points = points + adv_samples
    pred_class, _ = inference(classifier, adv_points)
    if is_sample:
        return adv_points, adv_target, pred_class, idx.squeeze().unsqueeze(-1).detach().cpu().numpy()
    else:
        return adv_points, adv_target, pred_class, None

