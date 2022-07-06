import numpy as np
import os
saved_dir="/home/jqf/Desktop/benchmark_pc_attack-master/baselines/attack_scripts/Exps/PointNet_num_points1024/All/GeoA3_0_BiStep1_IterStep10_Optadam_Lr0.01_Initcons10_CE_CDLoss1.0_HDLoss0.1_CurLoss1.0_k16"
objFilePath = os.path.join(saved_dir, 'PC')
files = os.listdir(objFilePath)
num = len(files)
all_point = []
points = []
all_adv_pc = []
all_real_lbl = []
all_target_lbl = []
for i in range(num):
    points = []
    l_1 = []
    obj_path = objFilePath + "/" + files[i]
    with open(obj_path) as file:
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == "vt":
                break
        f_n = []
        f_n = files[i].split("_")
        for j in f_n:
            if 'gt' in j:
                j = j.replace('gt', '')
                all_real_lbl.append(np.array([j]))
                #all_real_lbl = np.array(np.array([j]))
            if "attack" in j:
                j = j.replace('attack', '')
                all_target_lbl.append(np.array([j]))
                #all_target_lbl = np.array(np.array([j]))
        all_adv_pc.append(points)

        # all_real_lbl=np.array(np.array([15]))
        # all_target_lbl=np.array(np.array([5]))
all_adv_pc=np.array(all_adv_pc)
all_real_lbl = np.concatenate(all_real_lbl, axis=0)  # [num_data]
all_target_lbl = np.concatenate(all_target_lbl, axis=0)
save_path = './attack/results/{}_{}/GEOA3'.\
format("mn40", "1024")
isExists=os.path.exists(os.path.join(save_path,'npz'))
if not isExists:
    os.makedirs(os.path.join(save_path,'npz'))
np.savez(os.path.join(save_path, 'npz', 'GEO.npz'),
         test_pc=all_adv_pc.astype(np.float32),
         test_label=all_real_lbl.astype(np.float32),
         target_label=all_target_lbl.astype(np.uint8))