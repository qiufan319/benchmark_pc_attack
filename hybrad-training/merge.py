import numpy as np
import os
data1=np.load('/baselines/ht/Drop.npz')['test_pc']
label1=np.load('/baselines/ht/Drop.npz')['test_label']
data2=np.load('/baselines/ht/pgd.npz')['test_pc']
label2=np.load('/baselines/ht/pgd.npz')['test_label']
data3=np.load('/baselines/ht/Add.npz')['test_pc']
label3=np.load('/baselines/ht/Add.npz')['test_label']

data_all=np.vstack((data1,data2,data3))
label_all=np.hstack((label1,label2,label3))

save_path = '/baselines/hybrid_trainig'
save_name='defense.npy'
np.savez(os.path.join(save_path, save_name),
     test_pc=data_all.astype(np.float32),
     test_label=label_all.astype(np.uint8),
     target_label=data_all.astype(np.uint8),
     ori_pc=label_all.astype(np.float32)
     )
c=1