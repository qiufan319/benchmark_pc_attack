from __future__ import print_function
import sys
import os
root_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
import open3d as o3d #do not import open3d befor torch
import argparse
import os
import csv
import h5py
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from attack.ISO.utils import progress_bar, adjust_lr_steep, log_row
from torch.autograd import Variable
from baselines.model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg
from dataset import ModelNet40Attack
from attack.ISO.transforms_3d import *

# from models.pointnet import PointNetCls, feature_transform_regularizer
# from models.pointnet2 import PointNet2ClsMsg
# from models.dgcnn import DGCNN
# from models.pointcnn import PointCNNCls
from config import BEST_WEIGHTS
import attack.ISO.isometry_init
import attack.ISO.thompson_sample as ts



def save_visual_points(points, rt_matrix, args, i, penalty):
    # points size B*N*3 np array
    if not os.path.isdir('visual_data'):
        os.mkdir('visual_data')
    fname = 'visual_data/%s_%s_%s'%(args.data, args.model, args.name)
    if not os.path.isdir(fname):
        os.mkdir(fname)
    points = points.transpose(2, 1)
    craft = torch.mm(points[0,:,:], rt_matrix.transpose(0, 1))
    points, craft = points.cpu().numpy(), craft.unsqueeze(0).cpu().numpy()
    if not os.path.isdir('visual_data'):
        os.mkdir('visual_data')
    pcd = o3d.geometry.PointCloud()   
    pcd.points = o3d.utility.Vector3dVector(points[0,:,:])
    o3d.io.write_point_cloud('%s/num%s_p%2.4f_%s.ply' %(fname, i, penalty, 'origi'), pcd)

    pcd.points = o3d.utility.Vector3dVector(craft[0,:,:])
    o3d.io.write_point_cloud('%s/num%s_p%2.4f_%s.ply' %(fname, i, penalty, 'craft'), pcd)




def spectral_penalty(W, iters = 30):
    # power iteration method to get spectral norm of W^TW-I
    v = F.normalize(torch.empty(3).normal_(0,1), dim = 0, eps=1e-12).to(device)
    matrix = torch.mm(W.t(), W) - torch.eye(3).to(device)
    for _ in range(iters):
        v =  F.normalize(torch.matmul(matrix, v), dim = 0, eps=1e-12).to(device) 
        penalty = torch.dot(v, torch.matmul(matrix, v))
    return penalty.abs()

def iso_penalty(W, p=2):
    ## return the penalty of W away from rotation matrix 
    ## Schatten p-norm 
    matrix = torch.mm(W.t(), W) - torch.eye(3).to(device)
    penalty = (torch.mm(matrix.t(), matrix)).pow(p/2.).trace().pow(1./p)
    return penalty.abs()


def logits_info_2(obj, label, model):
    correct = 0
    logits, data = model(obj)
    logits=logits[0]
    prob = F.softmax(logits, dim=1)

    rates, indices = prob.sort(1, descending=True)
    rates, indices = rates.squeeze(0), indices.squeeze(0)

    correct += indices[0].eq(label.data).cpu().sum()

    return logits, correct.item(), rates, indices,data


def logits_info(obj, label, model,model_F='false'):
    correct = 0
    if model_F == 'false':
        logits, _ = model(obj)
        logits = logits[0]
    else:
        logits, _, _ = model(obj)

    prob = F.softmax(logits, dim=1)

    rates, indices = prob.sort(1, descending=True) 
    rates, indices = rates.squeeze(0), indices.squeeze(0)  

    correct += indices[0].eq(label.data).cpu().sum()
    
    return logits, correct.item(), rates, indices


class ISOnet(nn.Module):
    def __init__(self, model):
        super(ISOnet, self).__init__()
        self.model = model
        for p in self.parameters():
            p.requires_grad = False

        self.iso = nn.Linear(3, 3, bias = False)

    def forward(self, x):
        x_1 = self.iso(x.transpose(2,1)) #to B N 3
        x = self.model(x_1.transpose(2,1))#to B 3 N
        return x,x_1


def thompson_sample_attack(thompson, obj, label, model, num_init = 1):
    accuracies = []
    matrices = np.zeros([num_init,3,3])
    for i in range (num_init):
        arm = thompson.get_action() # arm corresponds to the 3 intervals of sampling
        reward, matrix = thompson.get_reward_matrix(arm, obj, label, model)
        model.iso.weight.data = torch.Tensor(matrix).to(device)
        _, correct, rates, indices = logits_info(obj, label, model)
        indx_label = (indices == label.item()).nonzero().item()
        accuracies.append(rates[indx_label].item())
        matrices[i,:,:] = matrix

        if reward == 1:
            break
    return torch.Tensor(matrices[np.argmin(accuracies),:,:]).to(device), thompson




def gradient_attack(obj, label, model, args):
    model.eval() #fix batchnorm layer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.step_size, betas=(0.9, 0.999))
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.step_size)
    penalty = torch.Tensor([0])
    for step in range(args.num_steps):
        optimizer.zero_grad()   
        

        logits, correct, rates, indices = logits_info(obj, label, model,model_F='false')
                

        indx_label = (indices == label.item()).nonzero().item()        
        if correct == 0:
            penalty = spectral_penalty(model.iso.weight.data)
            break

        if args.target == 0:
            loss1 = - F.cross_entropy(logits, label) # we want to maximize the loss
        else:
            _, _, _, indices = logits_info(obj, label, model)
            if indices[0] == label:  # classify is correct
                target_label = indices[1] # choose the 2nd largest as targeted label
                loss1 = torch.clamp(logits[0, indices[0]] - logits[0, target_label], min = -args.kappa)
            else:
                target_label = indices[0]
                loss1 = torch.clamp(logits[0, indices[1]] - logits[0, target_label], min = -args.kappa)
                # in fact loss1 is -kappa in this case            
        
        penalty = spectral_penalty(model.iso.weight.data).requires_grad_()
        loss2 = args.LAMBDA * penalty
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        



        progress_bar(step, args.num_steps, 
                     'Loss1: %.6f | Penalty: %.6f | True %d, %.3f%% | Cls %d, %.3f%%'
                     % (loss1, penalty, label, 100.*rates[indx_label], indices[0], 100.*rates[0]))

    return  correct, rates, indices, model, penalty.item(), step + 1





def recover(model, test_loader): 
    print('======> Recover the victim model') 
    model.iso.weight.data = torch.eye(3).to(device)
    test_loss, test_acc = test(test_loader, model)
    print('Recovered!')
    return model



def gen_attack_log(args):
    if not os.path.isdir('logs_attack'):
        os.mkdir('logs_attack')
    logname = ('logs_attack/ctri_%s_%s_%s.csv'%(args.data, args.model, args.name))
    
    if os.path.exists(logname):
        with open(logname, 'a') as logfile:
            log_row(logname, [''])
            log_row(logname, [''])
    
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['model type', 'data set', 'random seed',  
                'number of points in one batch', 'number of points in one object', 'model load path', 'steps of gradient-like attack', 
                'step size (lr) fo gradient-like attack', 'number of test objects', 'penalty coefficient',
                'target or not', 'kappa for CW', 'number of repeat initial attacks', 'number of divisions for theta', 'range of angle'])
        logwriter.writerow([args.model, args.data, args.seed, args.attack_batch_size, 
                args.num_points, args.model_path, args.num_steps, args.step_size,
                args.num_tests, args.LAMBDA, args.target, args.kappa, args.num_init, 
                args.d, args.a, args.b])
        logwriter.writerow(['Note', args.note])
        logwriter.writerow([''])

    return logname

def test(test_loader, model):
    model.eval()
    correct = 0
    total = 0
    for j, data in enumerate(test_loader, 0):
        points, label = data
        points, label = points.to(device), label.to(device)[:,0]
        points = points.transpose(2, 1)
        points, label = points.cuda(), label.cuda()
        logits,_ ,_= model(points)
        loss = F.cross_entropy(logits, label)
        logits_choice = logits.data.max(1)[1]
        correct += logits_choice.eq(label.data).cpu().sum()
        total += label.size(0)
        progress_bar(j, len(test_loader), 'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                    % (loss.item()/(j+1), 100.*correct.item()/total, correct, total))
        
    return loss.item()/(j+1), 100.*correct.item()/total



def log_thompson(logname, thompson):
    theta = thompson.alpha/(thompson.alpha + thompson.beta)
    log_row(logname, ['sample region infomation'])
    indx = np.argsort(-theta, axis=None)
    log_row(logname, -np.sort(-theta, axis=None).reshape(-1))
    log_row(logname, indx)
    print(-np.sort(-theta, axis=None).reshape(-1)[0:9])
    print(indx[0:9])
    
    log_row(logname, ['to axis'])
    axis_indx = np.unravel_index(np.argsort(-theta, axis=None), theta.shape)
    log_row(logname, axis_indx[0])
    log_row(logname, axis_indx[1])
    log_row(logname, axis_indx[2])
    print(axis_indx[0][0:9])
    print(axis_indx[1][0:9])
    print(axis_indx[2][0:9])
def write_h5(data, data_orig, label, label_orig,j):
    for i in range(data.shape[0]):
        h5_filename = 'baselines/attack_scripts/attack/results/mn40_1024/LGGAN/'+str(j)+'_'+str(i)+'.h5'
        h5f = h5py.File(h5_filename, 'w')
        if data.shape[-1]!=3:
            data=data.transpose(0,2,1)
        h5f.create_dataset('data', data=data[i,:,:])
        h5f.create_dataset('orig_data', data=data_orig[i,:,:].detach().cpu().numpy())
        h5f.create_dataset('label', data=label[i].cpu().numpy())
        h5f.create_dataset('orig_label', data=label_orig[i].cpu().numpy())
        h5f.close()
def log_penalty(logname, i, penalties, iso_penalties):
    log_row(logname, ['Max penalty', 'Average penalty in total', 'Variance in total', 
            'Average penalty over nonzeros', 'Variance over nonzeros'])
    if iso_penalties != []:
        log_row(logname, [max(penalties), 1.*sum(penalties)/(i+1), np.var(penalties),
             1.*sum(iso_penalties)/(len(iso_penalties)), np.var(iso_penalties)])
        print(max(penalties), 1.*sum(penalties)/(i+1), np.var(penalties),
             1.*sum(iso_penalties)/(len(iso_penalties)), np.var(iso_penalties))
    else:
        log_row(logname, [max(penalties), 1.*sum(penalties)/(i+1), np.var(penalties),'/', '/' ])
        print(max(penalties), 1.*sum(penalties)/(i+1), np.var(penalties),'/', '/')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='pointnet', help='choose victim model type')
    parser.add_argument('--data', type=str, default='mn40', help='choose vanila data set')
    parser.add_argument('--seed', type=int, default=0, help='manual random seed')
    parser.add_argument('--attack_batch_size', type=int, default=1, help='attack batch size')
    parser.add_argument('--num_points', type=int, default=1024, help='number of points sampled for one object')
    parser.add_argument('--model_path', default='example', help='path to store model state_dict') 
    parser.add_argument('--name', type=str, default='attack', help='name of the experiment')
    parser.add_argument('--num_steps', type=int, default=50, help='steps for gradient-like attack')
    parser.add_argument('--step_size', type=float, default=0.0005, help='learning rate for gradient-like attack')
    parser.add_argument('--num_tests', type=int, default=2000, help='number of tests for single attack')
    parser.add_argument('--LAMBDA', type=int, default=1000, help='coefficient for penalty')
    parser.add_argument('--target', type=int, default=1, help='whether use a targeted label or not (CW/CE loss), 0 means no target')
    parser.add_argument('--kappa', type=int, default=0, help='parameter for CW loss function')
    parser.add_argument('--num_init', type=int, default=50, help='number of repeat random initial attacks')
    parser.add_argument('--d', type=int, default=4, help='number of divisions for theta')
    parser.add_argument('--attack_type', type=str, default='combine', help='attack types')
    parser.add_argument('--note', type=str, default='', help='notation of the experiment')
    parser.add_argument('--feature_transform',type=int, default=False, help="use feature transform")
    parser.add_argument('--a', type=float, default=-np.pi, help='range of angle')
    parser.add_argument('--b', type=float, default=np.pi, help='range of angle')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--data_root', type=str,
                        default='../data/attack_data.npz')
    args = parser.parse_args()
    args.feature_transform = bool(args.feature_transform)
    args.seed = args.seed if args.seed > 0 else random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    torch.manual_seed(args.seed)
    BEST_WEIGHTS = BEST_WEIGHTS[args.data][args.num_points]
    ############################################################
    ## Load victim model
    ############################################################
    device='cuda'
    print('=====> Loading victim model from checkpoint...')
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
    model.eval()
    print('Successfully loaded!')
    # print('Stored test accuracy %.3f%%' % checkpoint['acc_list'][-1])
    data_set = ModelNet40Attack(args.data_root, num_points=args.num_points,
                                normalize=True)
    # test_sampler = DistributedSampler(test_set, shuffle=False)
    data_loader = DataLoader(data_set, batch_size=args.attack_batch_size,
                             shuffle=False, num_workers=0,
                             pin_memory=True, drop_last=False)
    # logname = gen_attack_log(args)

    it = iter(data_loader)
    corrects = []
    penalties = []
    iso_penalties = []

    # log_row(logname, ['Test number', 'True label', 'Prob before', 'Prob after',
    #     'Classification label', 'Prob after', 'Penalty', 'steps taken', 'Success Rate', 'Inital Rate'])
    init_suc = 0
    attack_times = 0

    
    thompson = ts.BernThompson(ts.environment(d = args.d, a0=args.a, b0=args.b))
    
    save_times = 0
    all_adv_pc = []
    all_real_lbl = []
    all_target_lbl = []
    for i in range(args.num_tests):     
        obj, label, target_label = next(it)
        obj, label = obj.to(device), label.to(device)

        obj = obj.transpose(2, 1)
    
        _, correct, rates, indices = logits_info(obj, label, model.to(device),model_F='true')
        indx_label = (indices == label.item()).nonzero().item()
        true_prob_before = 100.*rates[indx_label].item()
        steps = 0
        penalty = 0
        

        if correct == 0:
            pass
            #model fail, no need to attack
        else:
            attack_times += 1
            model_v = ISOnet(model = model).to(device)
            model_v.eval()
            
            model_v.iso.weight.data, thompson = thompson_sample_attack(thompson, obj, label, model_v, args.num_init)
            _, correct, rates, indices,data = logits_info_2(obj, label, model_v)
            indx_label = (indices == label.item()).nonzero().item()
            if correct == 0:
                #TSI attack success
                init_suc += 1
                adv_pc=data.detach().cpu().clone().numpy()
                all_adv_pc.append(adv_pc)
                all_real_lbl.append(label)
                all_target_lbl.append(target_label)
            elif args.attack_type == 'combine':            
                correct, rates, indices, model_v, penalty, steps = gradient_attack(obj, label, model_v, args)
                indx_label = (indices == label.item()).nonzero().item()
                if correct == 1:
                    penalty = 0 # CTRI attack fail
                    
                else:
                    iso_penalties.append(penalty)

            corrects.append(correct)
            penalties.append(penalty)

                
        
            if (correct == 0) and (true_prob_before>0.98 and rates[0].item()>0.98 or penalty < 0.05) and (save_times <= 10):
                save_times += 1
            
            # log_row(logname, [i, label.item(), true_prob_before, 100.*rates[indx_label].item(),
            #     indices[0].item(), 100.*rates[0].item(), penalty, steps,
            #     100.*(attack_times-sum(corrects))/attack_times, 100.*init_suc/attack_times])
            if i % 100 == 0:
                print('Attack success rate %.3f%%(%d/%d)'%(100.*(attack_times-sum(corrects))/attack_times, 
                            attack_times-sum(corrects), attack_times))

    # log_penalty(logname, i, penalties, iso_penalties)
    # log_thompson(logname, thompson)

    #save result
    all_adv_pc = np.concatenate(all_adv_pc, axis=0)
    all_real_lbl = np.array(all_real_lbl)  # [num_data]
    all_target_lbl = np.array(all_target_lbl)
    save_path = 'results/{}_{}/ISO'. \
        format("mn40", "1024")
    isExists = os.path.exists(os.path.join(save_path, 'npz'))
    if not isExists:
        os.makedirs(os.path.join(save_path, 'npz'))
    np.savez(os.path.join(save_path, 'npz', 'ISO.npz'),
             test_pc=all_adv_pc.astype(np.float32),
             test_label=all_real_lbl.astype(np.float32),
             target_label=all_target_lbl.astype(np.uint8))
