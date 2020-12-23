In this code, the architecture is different from that of the original STGCN Code (https://github.com/open-mmlab/mmskeleton/tree/master/deprecated/origin_stgcn_repo).

The sequences are of the following dimension: TxVxC where:
 T is the number of time frames, V is the number of vertices and C is the number of features. The number of people in the time frame can be 1 or 2. In the original STGCN code, the graph convolution concept is applied to the joints alone. In this code, the time frames are also treated as nodes in a graph and thus every time frame has a graph (of joints) present in them. So all the each time frame is connected to some other time frame for a given data. Refer to the image given below. 

 ![alt text](https://github.com/hariharannatesh/Action-Recognition-using-Pytorch-Geometric/blob/master/expt_3/New%20approach%20modified.jpg "Solving two person case")

 The sequences are batched along the 'T' dimension and this is taken care by the [DataListLoader](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.DataListLoader) of Pytorch Geometric. The number of time frames vary for each data. 

The *X_sub* protocol is followed. So the following are the training, validation and testing ids:
+ Train ids: 1,2,4,5,8,9,13,15,16,19,25,28,31,34,38
+ Validation ids: 14,17,18,27,35
+ Test ids: 3,6,7,10,11,12,20,21,22,23,24,26,29,30,32,33,36,37,39,40


Different hyper parameters are used for each experiment. The details and the plots are present in the respective directories. 

Date       | Expt  | Epochs  | Train accuracy | Val accuracy | Stopping epoch | Val accuracy at Stop Epoch |
-----------|-------|---------|----------------|--------------|----------------|----------------------------|
18/12/2020 |4a     | 120     | 0.83           | 0.6002       | 52             | 0.607                      |
18/12/2020 |4b     | 120     | 0.65           | 0.52         | 77             | 0.509                      |
21/12/2020 |4c     | 120     | 0.72           | 0.61         | 88             | 0.61                       |
21/12/2020 |4d     | 120     | 0.803          | 0.616        | 83             | 0.62                       |
