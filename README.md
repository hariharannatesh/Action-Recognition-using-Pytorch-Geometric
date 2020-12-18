This repository is about implementing the paper **Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition** [Arxiv Preprint](https://arxiv.org/abs/1801.07455) using the Pytorch Geometric package. 
A series of experiments will be conducted to test the various methods and to see which method gives the best result. The dataset used is **NTU-60 RGBD**. 

## Experiments 
- [x] Experiment 1: The sequences are padded along the temporal dimension to a common length.
- [x] Experiment 2: The data is padded along the feature dimension.
- [x] Experiment 3: The data for 2 person is batched together along the temporal dimension such that both the graphs are treated separately for spatial and temporal convolution.
- [x] Experiment original stgcn: The original code is trained from scratch to observe the training and validation plots.


## Table

Date           |Experiment |Epochs  | Training Accuracy | Validation accuracy
---------------|-----------|--------|-------------------|--------------------
7/12/2020      |Exp 1      | 80     |     0.23          |  0.016
10/12/2020     |Exp 2      | 80     |     0.66          |  0.57
14/12/2020     |Exp 3      | 80     |     0.71          |  0.60
15/12/2020     |Exp 3b     | 80     |     0.79          |  0.51 
18/12/2020     |Exp original_stgcn|80|    0.9993        |  0.7751

## Packages Used
+ Python (3.6.8)
+ Pytorch (1.6.0)
+ Torch-Geometric (1.6.1)
+ Torch-Cluster (1.5.7)
+ Torch-Scatter (2.0.5)
+ Torch-Sparse (0.6.7)
+ Torch-Spline-conv (1.2.0)
 
