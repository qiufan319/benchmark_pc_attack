# Preliminary

This folder contains data_split.py, merge.py, and training.py for hybrid training. data_split.py is used to evenly split the MN40 dataset into three segments for different attacks (e.g., PGD, drop, and add). And you can use the Python files in the baseline to generate the adversarial samples. merge.py is used to merge adversarial point clouds generated from different attacks into a defense.npz file. Lastly, training.py is used to facilitate the hybrid training.

