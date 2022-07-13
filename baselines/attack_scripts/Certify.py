from __future__ import print_function
import argparse
import os.path as osp
import torch
import torch.nn.functional as F
import csv
from time import time
import datetime
import os
import math

dataset_choices = ['modelnet40','modelnet10','scanobjectnn']
model_choices = ['pointnet2','dgcnn','curvenet','pointnet']
certification_method_choices = ['RotationX','RotationY','RotationZ','RotationXZ','RotationXYZ','Translation','Shearing','Tapering','Twisting','Squeezing','Stretching','GaussianNoise','Affine','AffineNoTranslation'] 



parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--dataset", default='modelnet40',choices=dataset_choices, help="which dataset")
parser.add_argument("--data_dir", default='',help="where is the dataset (for example scanobject)")
parser.add_argument("--model", type=str, choices=model_choices, help="model name")
parser.add_argument('--num_points', type=int, default=1024,help='num of points to use in case of curvenet, default 1024 recommended')
parser.add_argument('--max_features', type=int, default=1024,help='max features in Pointnet inner layers')
parser.add_argument("--base_classifier_path", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("--certify_method", type=str, default='rotationZ', required=True, choices=certification_method_choices, help='type of certification for certification')
parser.add_argument("--sigma", type=float, help="noise hyperparameter")
parser.add_argument("--experiment_name", type=str, required=True,help='name of directory for saving results',default='scanobjectnnPointnet2RotationZ0.05')
parser.add_argument("--certify_batch_sz", type=int, default=128, help="cetify batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=1000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--chunks", type=int, default=1, help="how many chunks do we cut the test set into")
parser.add_argument("--num_chunk", type=int, default=0, help="which chunk to certify")
parser.add_argument('--uniform', action='store_true', default=False, help='certify with uniform distribution')

args = parser.parse_args()

if (args.certify_method[0:8] == 'rotation' or args.certify_method[0:8] == 'Rotation') and args.sigma > 1:
    args.sigma = 1
    print("sigma above 1 for rotations is redundant (1 means +-Pi radians), setting sigma=1")


# full path for output
args.basedir = os.path.join('output/certify',args.dataset,args.certify_method, args.experiment_name)

# Log path: verify existence of output_path dir, or create it
if not os.path.exists(args.basedir):
    os.makedirs(args.basedir, exist_ok=True)
if not os.path.exists('output/samples/gaussianNoise'):
    os.makedirs('output/samples/gaussianNoise', exist_ok=True)
if not os.path.exists('output/samples/rotation'):
    os.makedirs('output/samples/rotation', exist_ok=True)
if not os.path.exists('output/samples/translation'):
    os.makedirs('output/samples/translation', exist_ok=True)
if not os.path.exists('output/samples/shearing'):
    os.makedirs('output/samples/shearing', exist_ok=True)
if not os.path.exists('output/samples/tapering'):
    os.makedirs('output/samples/tapering', exist_ok=True)
if not os.path.exists('output/samples/twisting'):
    os.makedirs('output/samples/twisting', exist_ok=True)
if not os.path.exists('output/samples/squeezing'):
    os.makedirs('output/samples/squeezing', exist_ok=True)
if not os.path.exists('output/samples/stretching'):
    os.makedirs('output/samples/stretching', exist_ok=True)
if not os.path.exists('output/samples/affineNoTranslation'):
    os.makedirs('output/samples/affineNoTranslation', exist_ok=True)
if not os.path.exists('output/samples/affine'):
    os.makedirs('output/samples/affine', exist_ok=True)

args.outfile = os.path.join(args.basedir, 'certification_chunk_'+str(args.num_chunk+1)+'out_of'+str(args.chunks)+'.txt')


if __name__ == "__main__":

    #use cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    if args.model == 'pointnet2':
        from Pointnet2andDGCNN.Trainers.pointnet2Train import Net
        from torch_geometric.datasets import ModelNet
        import torch_geometric.transforms as T
        from torch_geometric.data import DataLoader
        from SmoothedClassifiers.Pointnet2andDGCNN.SmoothFlow import SmoothFlow
        from Pointnet2andDGCNN.DataLoaders import ScanobjectDataset

        if args.dataset == 'modelnet40':
            
            #dataset and loaders
            path = osp.join(osp.dirname(osp.realpath(__file__)), 'Data/PointNet2andDGCNN/Modelnet40fp')
            pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
            print(path)
            test_dataset = ModelNet(path, '40', False, transform, pre_transform)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)


            num_classes = 40

        elif args.dataset == 'modelnet10':
            
            #dataset and loaders
            path = osp.join(osp.dirname(osp.realpath(__file__)), 'Data/PointNet2andDGCNN/Modelnet10fp')
            pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
            print(path)
            test_dataset = ModelNet(path, '10', False, transform, pre_transform)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)


            num_classes = 10

        elif args.dataset == 'scanobjectnn':
            test_dataset = ScanobjectDataset.ScanObjectNN(args.data_dir, 'test',  args.num_points,
                                    variant='obj_only', dset_norm="inf")
            classes = test_dataset.classes
            num_classes = len(classes)

            test_loader = DataLoader(test_dataset, batch_size=1,
                                    shuffle=False, num_workers=0)
        
        #model and optimizer
        base_classifier = Net(num_classes).to(device)
        optimizer = torch.optim.Adam(base_classifier.parameters(), lr=0.001)

        #loadTrainedModel
        checkpoint = torch.load(args.base_classifier_path)
        base_classifier.load_state_dict(checkpoint['model_param'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    elif args.model == 'dgcnn':
        from Pointnet2andDGCNN.Trainers.dgcnnTrain import Net
        from torch_geometric.datasets import ModelNet
        import torch_geometric.transforms as T
        from torch_geometric.data import DataLoader
        from SmoothedClassifiers.Pointnet2andDGCNN.SmoothFlow import SmoothFlow
        from Pointnet2andDGCNN.DataLoaders import ScanobjectDataset

        if args.dataset == 'modelnet40':

            path = osp.join(osp.dirname(osp.realpath(__file__)), 'Data/PointNet2andDGCNN/Modelnet40fp')
            pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024) #convert to pointcloud
            print(path)
            test_dataset = ModelNet(path, '40', False, transform, pre_transform)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=0)

            num_classes = 40

        elif args.dataset == 'modelnet10':

            path = osp.join(osp.dirname(osp.realpath(__file__)), 'Data/PointNet2andDGCNN/Modelnet10fp')
            pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024) #convert to pointcloud
            print(path)
            test_dataset = ModelNet(path, '10', False, transform, pre_transform)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=0)

            num_classes = 10
        
        elif args.dataset == 'scanobjectnn':
            test_dataset = ScanobjectDataset.ScanObjectNN(args.data_dir, 'test',  args.num_points,
                                    variant='obj_only', dset_norm="inf")
            classes = test_dataset.classes
            num_classes = len(classes)

            test_loader = DataLoader(test_dataset, batch_size=1,
                                    shuffle=False, num_workers=0)
        
        #model and optimizer
        base_classifier = Net(num_classes, k=20).to(device)
        optimizer = torch.optim.Adam(base_classifier.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        #loadTrainedModel
        checkpoint = torch.load(args.base_classifier_path)
        base_classifier.load_state_dict(checkpoint['model_param'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    elif args.model == 'curvenet':
        
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
        from CurveNet.core.data import ModelNet40,ScanObjectNN,collate_fn
        from CurveNet.core.models.curvenet_cls import CurveNet
        import numpy as np
        from torch.utils.data import DataLoader
        from CurveNet.core.util import cal_loss, IOStream
        import sklearn.metrics as metrics
        from SmoothedClassifiers.CurveNetandPointnet.SmoothFlow import SmoothFlow

        if args.dataset == 'modelnet40':

            test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),batch_size=1, shuffle=False, drop_last=False)

            num_classes = 40

        elif args.dataset == 'modelnet10':
            raise NotImplementedError
        
        elif args.dataset == 'scanobjectnn':
            test_dataset = ScanObjectNN(args.data_dir, 'test',  args.num_points,
                                    variant='obj_only', dset_norm="inf")
            classes = test_dataset.classes
            num_classes = len(classes)

            test_loader = DataLoader(test_dataset, batch_size=1,
                                    shuffle=False, num_workers=0,collate_fn=collate_fn)
        
        #declare and load pretrained model
        base_classifier = CurveNet(num_classes=num_classes).to(device)
        base_classifier = nn.DataParallel(base_classifier)
        base_classifier.load_state_dict(torch.load(args.base_classifier_path))
        base_classifier.eval()
    
    elif args.model == 'pointnet':
        
        import sys
        sys.path.insert(0, osp.join(osp.dirname(osp.realpath(__file__)),'Pointnet'))
        #sys.path.insert(0, "/home/santamgp/Documents/CertifyingAffineTransformationsOnPointClouds/3D-RS-PointCloudCertifying/Pointnet")

        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        from Pointnet.DataLoaders import datasets
        from torch.utils.data import DataLoader
        from Pointnet.model import PointNet
        from SmoothedClassifiers.CurveNetandPointnet.SmoothFlow import SmoothFlow

        if args.dataset == 'modelnet40':
            
            test_data = datasets.modelnet40(num_points=args.num_points, split='test', rotate='none')

            test_loader = DataLoader(
                dataset=test_data,
                batch_size=1,
                shuffle=False,
                num_workers=0
            )
            
            num_classes = test_data.num_classes

            

        elif args.dataset == 'modelnet10':
            raise NotImplementedError
        
        elif args.dataset == 'scanobjectnn':
            test_dataset = datasets.ScanObjectNN(args.data_dir, 'test',  args.num_points,
                                    variant='obj_only', dset_norm="inf")
            classes = test_dataset.classes
            num_classes = len(classes)

            test_loader = DataLoader(test_dataset, batch_size=1,
                                    shuffle=False, num_workers=0,collate_fn=datasets.collate_fn)
                    
        base_classifier = PointNet(
                number_points=args.num_points,
                num_classes=num_classes,
                max_features=args.max_features,
                pool_function='max',
                transposed_input= True
            )
        base_classifier = base_classifier.to(device)

        objective = nn.CrossEntropyLoss()
        optimizer = optim.Adam(base_classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        #loadTrainedModel
        try:
            checkpoint = torch.load(args.base_classifier_path)
            base_classifier.load_state_dict(checkpoint['model_param'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        except:
            #before saying there is no model check if it is the 3d certify authors pretrained model
            try:
                base_classifier.load_state_dict(torch.load(args.base_classifier_path))
            except:
                print('no pretrained model found')
        
        base_classifier.eval()
        
    else:
        raise Exception("Undefined model!") 

       

    if args.certify_method[0:8] == 'rotation' or args.certify_method[0:8] == 'Rotation':
        args.sigma *= math.pi # For rotaions to transform the angles to [0, pi]
    # create the smooothed classifier g
    smoothed_classifier = SmoothFlow(base_classifier, num_classes, args.certify_method, args.sigma)

    # prepare output txt and csv files
    csvoutfile = os.path.join(args.basedir, 'certification_chunk_'+str(args.num_chunk+1)+'out_of'+str(args.chunks)+'.csv')
    ftxt = open(args.outfile, 'w')
    fcsv = open(csvoutfile, 'w')

    # create the csv writer
    writer = csv.writer(fcsv)

    #print training params
    d = vars(args)
    text = ' | '.join([str(key) + ': ' + str(d[key]) for key in d])
    print(text, file=ftxt)
    writer.writerow([str(key) + ': ' + str(d[key]) for key in d])

    #print header
    print("idx\t\tlabel\t\tpredict\t\tradius\t\tcorrect\t\ttime", file=ftxt, flush=True)
    writer.writerow(["idx","label","predict","radius","correct","time"])

    # iterate through the dataset
    dataset = [u for u in test_loader]

    interval = len(dataset)//args.chunks
    start_ind = args.num_chunk * interval

    #which pointcloud to take as sample in the output
    sampleNumber = 0
    
    for i in range(start_ind, start_ind + interval):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break
        
        #check if this is the pointcloud to sample
        if i == sampleNumber:
            plywrite = True
        else:
            plywrite = False


        #extract one at a time and the corresponding label
        x = dataset[i]
        if args.model == 'dgcnn' or args.model == 'pointnet2':
            label = x.y.item()
            x = x.to(device)
        elif args.model == 'curvenet':
            label = x[1].item()
            x[0] = x[0].to(device)
            x[1] = x[1].to(device)
        elif args.model == 'pointnet':
            label = x[2].item()
            x[1] = x[2]
            x[0] = x[0].to(device)
            x[1] = x[1].to(device)

        before_time = time()
        # certify the prediction of g around x
        
        prediction, radius, p_A = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.certify_batch_sz,plywrite)
        if args.uniform:
            radius = 2 * args.sigma * (p_A - 0.5)
        after_time = time()
        correct = int(prediction == label)
        print('Time spent certifying pointcloud {} was {} sec \t {}/{} ({:.2}%)'.format(i,after_time - before_time,i,start_ind + interval,100*i/(start_ind + interval)) )
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t\t{}\t\t{}\t\t{:.3}\t\t{}\t\t{}".format(i, label, prediction, radius, correct, time_elapsed), file=ftxt, flush=True)
        writer.writerow([i, label, prediction, radius, correct, time_elapsed])

    ftxt.close()
    fcsv.close()