# Benchmark of Pointcloud Adversarial Attacks and Defenses
## Introduction
Point cloud is an important 3D representation that is widely used in various security-critical applications. Although deep learning on point cloud has exhibited high performance, recent research has shown that deep learning models on point cloud are vulnerable to adversarial attacks. To help researchers more easily compare existing attacks and defenses, in this repo we provide PyTorch implementations for common baseline attacks and defenses.
All our code is modified or integrated from other existing code.
## Requirement

Our project is developed using Python 3.8, PyTorch 1.11.0 with CUDA11.6. We recommend you to use [anaconda](https://www.anaconda.com/) for dependency configuration.


First create an anaconda environment called ```benchmark``` by
Cancel changes
```shell
conda create -n benchmark python=3.8

conda activate benchmark
```

All the required packages are listed in the requirements.txt file. First use pip to install them by

```shell
python -m pip install -r requirements.txt
```

Then, you need to install torch, torchvision and torch-scatter manually to fit in with your server environment (e.g. CUDA version). For the torch and torchvision used in my project, run

```shell
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorchCancel changes
```
## Project Structure
The code in this project is mainly divided into two folders, each of which has a detailed README file explaining its usage. Here I will briefly outline the structure of this repo.\
```baselines/``` contains code for training/testing the victim models as well as some baseline attack and defense methods.\
```AT/``` contains code for adversarial training
```ConvONet/``` contains code for IF-defense
```pointcutmix/```contains code for pointcutmix
Please go to each folder to see how to use the code in detail.



## Result
Classification accuracy of ModelNet40 under black-box attacks and defense.\
***baslines***: PointNet

##  Acknowledgements
We thank the authors of following works for opening source their excellent codes.

- [PointNet/PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch), [DGCNN](https://github.com/WangYueFt/dgcnn), [PointConv](https://github.com/DylanWusee/pointconv_pytorch)
- [Perturb/Add attack](https://github.com/xiangchong1/3d-adv-pc), [kNN attack](https://github.com/jinyier/ai_pointnet_attack), [Drop attack](https://github.com/tianzheng4/PointCloud-Saliency-Maps)
- [PU-Net](https://github.com/lyqun/PU-Net_pytorch), [DUP-Net](https://github.com/RyanHangZhou/DUP-Net)
- [ONet](https://github.com/autonomousvision/occupancy_networks)
- [IF-Defense](https://github.com/Wuziyi616/IF-Defense)
- [GEOA3](https://github.com/Gorilla-Lab-SCUT/GeoA3)
