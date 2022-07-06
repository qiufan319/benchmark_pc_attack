import numpy as np
import os
# cat_data = np.load(r'C:\Users\JQF\Desktop\IF-Defense-main\IF-Defense-main\test.npz')
# h=1
# folder=r'F:\GeoA3-master\GeoA3-master\npz'
# files=os.listdir(folder)
# all_pc, all_lbl, all_target = [], [], []
# for file in files:
#     one_file = os.path.join(folder, file)
#     npz = np.load(one_file)
#     all_pc.append(npz['test_pc'])
#     all_lbl.append(npz['test_label'])
#     all_target.append(npz['target_label'])
# # concat data
# all_pc = np.concatenate(all_pc, axis=0)
# all_lbl = np.concatenate(all_lbl, axis=0)
# all_target = np.concatenate(all_target, axis=0)
# np.savez(os.path.join(folder, 'merge'),
#              test_pc=all_pc.astype(np.float32),
#              test_label=all_lbl.astype(np.uint8),
#              target_label=all_target.astype(np.uint8))

import scipy.io as io
io.savemat(r'F:\benchmark_pc_attack\baselines\data\test.mat', mdict=np.load(r'C:\Users\JQF\Desktop\IF-Defense-main\benchmark_pc_attack\baselines\data\MN40_random_2048.npz'))