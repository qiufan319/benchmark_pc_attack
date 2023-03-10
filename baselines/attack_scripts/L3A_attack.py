import sys
import os

import numpy as np
root_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path, 'model'))
sys.path.append(os.path.join(root_path, 'attack','L3A'))


from adv_utils import *
from tensorboardX import SummaryWriter
from shutil import copyfile
from model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.backends.cudnn.enabled = False
from dataset import ModelNet40Attack
from config import BEST_WEIGHTS

BEST_WEIGHTS = BEST_WEIGHTS['mn40'][1024]
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
    all_ori_pc=[]
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
        # visualization

        #save result
        all_adv_pc.append(adv_points.detach().cpu().numpy())
        all_real_lbl.append(target.detach().cpu().numpy())
        all_ori_pc.append(points.detach().cpu().numpy())
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
    all_ori_pc=np.concatenate(all_ori_pc,axis=0)
    # save results
    save_path = 'baselines/result'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = 'L3A_result.npz'
    np.savez(os.path.join(save_path, save_name),
             test_pc=all_adv_pc.astype(np.float32),
             test_label=all_real_lbl.astype(np.uint8),
             target_label=all_target_lbl.astype(np.uint8),
             ori_pc=all_ori_pc.astype(np.float32)
             )

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    class_acc_a[:, 2] = class_acc_a[:, 0] / class_acc_a[:, 1]
    class_acc_a = np.mean(class_acc_a[:, 2])
    instance_acc_a = np.mean(mean_correct_a)

    return class_acc, instance_acc, class_acc_a, instance_acc_a


if __name__ == '__main__':
    base_para = {
        'model' : 'dgcnn',
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
        'data_root':'baselines/data/attack_data.npz',
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
    MODEL_loss = importlib.import_module('pointnet_cls')

    model = MODEL_loss.get_model(num_class, normal_channel=False).cuda()
    floss = MODEL_loss.get_loss().cuda()

    # build model
    if para['model'].lower() == 'dgcnn':
        model = DGCNN(1024, 20, output_channels=40).cuda()
    elif para['model'].lower() == 'pointnet_cls':
        model = PointNetCls(k=40, feature_transform=True).cuda()
    elif para['model'].lower() == 'pointnet2':
        model = PointNet2ClsSsg(num_classes=40)
    elif para['model'].lower() == 'pointconv':
        model = PointConvDensityClsSsg(num_classes=40).cuda()
    else:
        print('Model not recognized')
        exit(-1)

    # load model weight
    state_dict = torch.load(
        BEST_WEIGHTS[para['model']], map_location='cpu')
    print('Loading weight {}'.format(BEST_WEIGHTS[para['model']]))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # eliminate 'module.' in keys
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    # checkpoint = torch.load('baselines/pretrain/pointnet_cls.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])


    #Load data
    DATA_PATH = para['data_root']
    # TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=1024, split=para['split_name'], normal_channel=False)
    # data_loader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=12, shuffle=False, num_workers=4)
    TEST_DATASET = ModelNet40Attack(DATA_PATH, num_points=1024,
                                normalize=False)
    data_loader =torch.utils.data.DataLoader(TEST_DATASET, batch_size=para['batch_size'],
                             shuffle=False, num_workers=0,
                             pin_memory=True, drop_last=False)


    class_acc, instance_acc, class_acc_a, instance_acc_a = test_attack(model, data_loader, floss, para)

    print('before Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
    print('after Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc_a, class_acc_a))

