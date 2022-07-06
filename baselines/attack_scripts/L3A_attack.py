"""
Author: Sun
Date: April 2021
"""
import sys
import os

import numpy as np

root_path=os.path.abspath(os.path.join(os.getcwd(), "../"))
sys.path.append(os.path.join(root_path, 'model'))
root_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from attack.L3A.adv_utils import *
from tensorboardX import SummaryWriter
from shutil import copyfile
# from model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.backends.cudnn.enabled = False
from dataset import ModelNet40Attack

def test_attack(model, loader, criterion, para):
    adv_t = para['adv_target_idx']
    save_pn_file = para['save_pn_file']
    file_idx = np.ones(40, dtype=int)
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    mean_correct_a = []
    class_acc_a = np.zeros((num_class, 3))
    all_adv_pc = []
    all_real_lbl = []
    all_target_lbl = []
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        pred_names = []
        adv_names = []
        print("BATCH: %d" % j)
        points, target ,t_L= data
        # target = target[:, 0]
        # points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        with torch.no_grad():
            pred_choice, _ = inference(model, points)
            ori_names = []
            for i in range(points.shape[0]):
                 ori_names.append(cls_names[target[i]])

            for cat in np.unique(target.cpu()):
                classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
                class_acc[cat, 1] += 1
            correct = pred_choice.eq(target.long().data).cpu().sum()
            if adv_t is not None:
                misjudge = pred_choice.eq(adv_target.long().data).cpu().sum()
                print('before ori accuracy:%f, adv accuracy:%f' 
                      % ((correct.item() / float(points.size()[0])), (misjudge.item() / float(points.size()[0]))))
            else:
                print('before ori accuracy:%f' % (correct.item() / float(points.size()[0])))
            mean_correct.append(correct.item() / float(points.size()[0]))

        adv_points, adv_target, pred_class, _ = adv_propagation(model.eval(), criterion, points, target, para)

        #save result
        all_adv_pc.append(adv_points.detach().cpu().numpy())
        all_real_lbl.append(target.detach().cpu().numpy())
        all_target_lbl.append(t_L)

        for k in range(points.shape[0]):
            pred_names.append(cls_names[pred_class[k]])
            adv_names.append(cls_names[adv_target[k]])
        print('ORI_TARGET:{}\nADV_TARGET:{}\nPRED_RESULT:{}'.format(ori_names, adv_names, pred_names))
        with torch.no_grad():
            pred_choice_a, _ = inference(model.eval(), adv_points)
            for cat in np.unique(target.cpu()):
                classacc_a = pred_choice_a[target == cat].eq(target[target == cat].long().data).cpu().sum()
                class_acc_a[cat, 0] += classacc_a.item() / float(adv_points[target == cat].size()[0])
                class_acc_a[cat, 1] += 1
            correct_a = pred_choice_a.eq(target.long().data).cpu().sum()
            misjudge_a = pred_choice_a.eq(adv_target.long().data).cpu().sum()
            print('after ori accuracy:%f, adv accuracy:%f'
                  % ((correct_a.item() / float(points.size()[0])), (misjudge_a.item() / float(points.size()[0]))))
            mean_correct_a.append(correct_a.item() / float(adv_points.size()[0]))
    all_adv_pc = np.concatenate(all_adv_pc, axis=0)
    all_real_lbl = np.concatenate(all_real_lbl, axis=0)  # [num_data]
    all_target_lbl = np.concatenate(all_target_lbl, axis=0)
    # save results
    save_path = 'results/mn40_1024/L3A'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = 'L3A_result.npz'
    np.savez(os.path.join(save_path, save_name),
             test_pc=all_adv_pc.astype(np.float32),
             test_label=all_real_lbl.astype(np.uint8),
             target_label=all_target_lbl.astype(np.uint8))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    class_acc_a[:, 2] = class_acc_a[:, 0] / class_acc_a[:, 1]
    class_acc_a = np.mean(class_acc_a[:, 2])
    instance_acc_a = np.mean(mean_correct_a)

    return class_acc, instance_acc, class_acc_a, instance_acc_a


if __name__ == '__main__':
    base_para = {
        'model' : 'pointnet',
        'split_name' : 'test',
        'step_num' : 200,
        'lmd' : 0.2,
        'dloss' : 'L2',
        'is_sample' : True,
        'n_points' : 50,
        'n_samples' : 50,
        'radius' : 0.25,
        'back_thr' : 0.1,
        'is_specific' : True,
        'adv_target_idx' : None,
        'save_pn_file' : True,
        'save_as_dataset' : True,
        'is_pwa' : False,
        'is_lcons' : False,
        'data_root':'../data/attack_data.npz',
        'batch_size':12
    }

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = BASE_DIR
    sys.path.append(os.path.join(ROOT_DIR, 'models'))

    para = base_para

    experiment_dir = 'log/classification/' + para['model']
    #Load model
    num_class = 40
    # model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module('pointnet_cls')

    model = MODEL.get_model(num_class, normal_channel=False).cuda()

    checkpoint = torch.load('../pretrain/pointnet_cls.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    floss = MODEL.get_loss().cuda()

    #Load data
    DATA_PATH = para['data_root']
    # TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=1024, split=para['split_name'], normal_channel=False)
    # data_loader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=12, shuffle=False, num_workers=4)
    TEST_DATASET = ModelNet40Attack(DATA_PATH, num_points=1024,
                                normalize=False)
    data_loader =torch.utils.data.DataLoader(TEST_DATASET, batch_size=para['batch_size'],
                             shuffle=False, num_workers=0,
                             pin_memory=True, drop_last=False)
    # if para['is_sample']:
    #     print(para['model'] + ' lmd:' + str(para['lmd']) + ' ' + para['dloss'])
    # else:
    #     print(para['model'] + ' lmd:' + str(para['lmd']) + ' ' + para['dloss'] + ' globel')
    #
    # print("N_POINTS = {}, N_SAMPLES = {}, RADIUS = {}, BACK_THR = {}\n"
    #     .format(para['n_points'], para['n_samples'], para['radius'], para['back_thr']))

    class_acc, instance_acc, class_acc_a, instance_acc_a = test_attack(model, data_loader, floss, para)

    print('before Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
    print('after Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc_a, class_acc_a))

