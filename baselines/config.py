"""Config file for automatic code running
Assign some hyper-parameters, e.g. batch size for attack
"""
import os
import sys
root_path=os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(os.path.join(root_path))
BEST_WEIGHTS = {
    # trained on standard mn40 dataset
    'mn40': {
        1024: {
            'pointnet': 'baselines/pretrain/mn40/pointnet.pth',
            'pointnet2': 'baselines/pretrain/mn40/pointnet2.pth',
            'pointconv': 'baselines/pretrain/mn40/pointconv.pth',
            'dgcnn': 'baselines/pretrain/mn40/dgcnn.pth',
            'curvenet': 'baselines/pretrain/mn40/curvenet.pth',
            'pct': 'baselines/pretrain/mn40/pct.pth',
            'gda': 'baselines/pretrain/mn40/gda.pth',
            'rpc': 'baselines/pretrain/mn40/rpc.pth'
        },

    },
    #trained on mn40 + PGD AT
    'AT_mn40': {
            1024: {
                'pointnet': 'baselines/pretrain/AT/pointnet.pth',
                'pointnet2': 'baselines/pretrain/AT/pointnet2.pth',
                'pointconv': 'baselines/pretrain/AT/pointconv.pth',
                'dgcnn': 'baselines/pretrain/AT/dgcnn.pth',
                'curvenet': 'baselines/pretrain/AT/curvenet.pth',
                'pct': 'baselines/pretrain/AT/pct.pth',
                'gda': 'baselines/pretrain/AT/gda.pth',
                'rpc': 'baselines/pretrain/AT/rpc.pth',
            },

        },
    # trained on mn40 + cutmix
    'cutmix': {
        1024: {
            'pointnet': 'baselines/pretrain/pointcutmix/pointnet.pth',
                'pointnet2': 'baselines/pretrain/pointcutmix/pointnet2.pth',
                'pointconv': 'baselines/pretrain/pointcutmix/pointconv.pth',
                'dgcnn': 'baselines/pretrain/pointcutmix/dgcnn.pth',
                'curvenet': 'baselines/pretrain/pointcutmix/curvenet.pth',
                'pct': 'baselines/pretrain/pointcutmix/pct.pth',
                'gda': 'baselines/pretrain/pointcutmix/gda.pth',
                'rpc': 'baselines/pretrain/pointcutmix/rpc.pth',
        },
    },
    # trained on mn40 + ONet optimized mn40
    'hybrid_training': {
        1024: {
            'pointnet': 'baselines/pretrain/hybrid_training/pointnet.pth',
                'pointnet2': 'baselines/pretrain/hybrid_training/pointnet2.pth',
                'pointconv': 'baselines/pretrain/hybrid_training/pointconv.pth',
                'dgcnn': 'baselines/pretrain/hybrid_training/dgcnn.pth',
                'curvenet': 'baselines/pretrain/hybrid_training/curvenet.pth',
                'pct': 'baselines/pretrain/hybrid_training/pct.pth',
                'gda': 'baselines/pretrain/hybrid_training/gda.pth',
                'rpc': 'baselines/pretrain/hybrid_training/rpc.pth',
        },
    },
    # trained on mn40 + ConvONet optimized mn40
}

# PU-Net trained on Visionair with 1024 input point number, up rate 4
PU_NET_WEIGHT = 'baselines/defense/DUP_Net/pu-in_1024-up_4.pth'

# Note: the following batch sizes are tested on a RTX 2080 Ti GPU
# you may need to slightly adjust them to fit in your device

# max batch size used in testing model accuracy
MAX_TEST_BATCH = {
    1024: {
        'pointnet': 512,
        'pointnet2': 256,
        'dgcnn': 96,
        'pointconv': 320,
    },
}

# max batch size used in testing model accuracy with DUP-Net defense
# since there will be 4x points in DUP-Net defense results
MAX_DUP_TEST_BATCH = {
    1024: {
        'pointnet': 160,
        'pointnet2': 80,
        'dgcnn': 26,
        'pointconv': 48,
    },
}

# max batch size used in Perturb attack
MAX_PERTURB_BATCH = {
    1024: {
        'pointnet': 384,
        'pointnet2': 78,
        'dgcnn': 52,
        'pointconv': 57,
    },
}

# max batch size used in kNN attack
MAX_KNN_BATCH = {
    1024: {
        'pointnet': 248,
        'pointnet2': 74,
        'dgcnn': 42,
        'pointconv': 54,
    },
}

# max batch size used in Add attack
MAX_ADD_BATCH = {
    1024: {
        'pointnet': 256,
        'pointnet2': 78,
        'dgcnn': 35,
        'pointconv': 57,
    },
}

# max batch size used in Add Cluster attack
MAX_ADD_CLUSTER_BATCH = {
    1024: {
        'pointnet': 320,
        'pointnet2': 88,
        'dgcnn': 45,
        'pointconv': 60,
    },
}

# max batch size used in Add Object attack
MAX_ADD_OBJECT_BATCH = {
    1024: {
        'pointnet': 320,
        'pointnet2': 88,
        'dgcnn': 42,
        'pointconv': 58,
    },
}

# max batch size used in Drop attack
MAX_DROP_BATCH = {
    1024: {
        'pointnet': 360,
        'pointnet2': 80,
        'dgcnn': 52,
        'pointconv': 57,
    },
}

MAX_FGM_PERTURB_BATCH = {
    1024: {
        'pointnet': 360,
        'pointnet2': 76,
        'dgcnn': 52,
        'pointconv': 58,
    },
}
