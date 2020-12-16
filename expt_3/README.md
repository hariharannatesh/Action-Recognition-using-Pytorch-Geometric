In this code, the architecture is different from that of the original STGCN Code (https://github.com/open-mmlab/mmskeleton/tree/master/deprecated/origin_stgcn_repo).

The sequences are of the following dimension: TxVxC where:
 T is the number of time frames, V is the number of vertices and C is the number of features. The number of people in the time frame can be 1 or 2. In the original STGCN code, the graph convolution concept is applied to the joints alone. In this code, the time frames are also treated as nodes in a graph and thus every time frame has a graph (of joints) present in them. So all the each time frame is connected to some other time frame for a given data. Refer to the image given below. 

 ![alt text](https://github.com/hariharannatesh/Action-Recognition-using-Pytorch-Geometric/blob/master/expt_3/New%20approach%20modified.jpg "Solving two person case")

 The sequences are batched along the 'T' dimension and this is taken care by the [DataListLoader](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.DataListLoader) of Pytorch Geometric. The number of time frames vary for each data. 

The *X_sub* protocol is followed. So the following are the training, validation and testing ids:
+ Train ids: 1,2,4,5,8,9,13,15,16,19,25,28,31,34,38
+ Validation ids: 14,17,18,27,35
+ Test ids: 3,6,7,10,11,12,20,21,22,23,24,26,29,30,32,33,36,37,39,40

Date of Experiment: 14 - 12 - 2020

The following hyper parameters are used:

 Optimizer  | Learning Rate |  Momentum    |  Batch Size | Epochs
 ------------- | -------------| ---------- | ---------| -------
 SGD           | 0.1          | 0.9        | 64   | 80 


+ The non-linear function used is ReLu. 
+ Dropout value used is 0.5. 
+ Number of STGCN Blocks used is 10.
+ Gnode used: gnode39
+ Number of GPUs used: 4
+ Edge importance parameter is set to True.

After 80 epochs, the training and validation plot: https://colab.research.google.com/drive/1UL-qh4RUqC68faO2Kef6WYXgCYu5aNvc#scrollTo=8fgQaR6v6C6x

## Expt 3b:

Now a learning rate scheduler is added:
scheduler=lr_scheduler.MultiStepLR(optimizer,milestones=[12,52],gamma=0.1)

The other parameters are kept same and the model is trained again.


After 80 epochs, the training and validation plot: https://colab.research.google.com/drive/1oXSEQPzokl58c6fU-EgMwjYhoyQAbCXj#scrollTo=8fgQaR6v6C6x