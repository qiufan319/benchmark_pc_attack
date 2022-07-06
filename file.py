import os
import numpy as np

objFilePath = r'F:\GeoA3-master\GeoA3-master\Exps\PointNet_npoint1024\All\GeoA3_0_BiStep10_IterStep500_Optadam_Lr0.01_Initcons10_CE_CDLoss1.0_HDLoss0.1_CurLoss1.0_k16\PC'
files=os.listdir(objFilePath)
num=len(files)
all_point=[]
for i in range(num):
    points = []
    all_adv_pc = []
    all_real_lbl = []
    all_target_lbl = []
    l_1=[]
    obj_path=objFilePath+"\\"+files[i]
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
        f_n=[]
        f_n=files[i].split("_")
        for j in f_n:
            if 'gt' in j:
                j=j.replace('gt','')
                all_real_lbl.append(np.array([j]))
                all_real_lbl = np.array(np.array([j]))
            if "attack" in j:
                j=j.replace('attack','')
                all_target_lbl.append(np.array([j]))
                all_target_lbl = np.array(np.array([j]))
        all_adv_pc.append(points)
        all_adv_pc=np.array(all_adv_pc)
        # all_real_lbl=np.array(np.array([15]))
        # all_target_lbl=np.array(np.array([5]))
        np.savez(os.path.join(r'F:\GeoA3-master\GeoA3-master\npz\test'+str(i)),
                 test_pc=all_adv_pc.astype(np.float32),
                 test_label=all_real_lbl.astype(np.uint8),
                 target_label=all_target_lbl.astype(np.uint8))
        #points = np.array(points)
# points原本为列表，需要转变为矩阵，方便处理

#np.save(r'C:\Users\JQF\Desktop\IF-Defense-main\IF-Defense-main\test.npy',points)