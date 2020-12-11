In this code, the architecture is different from that of the original STGCN Code (https://github.com/open-mmlab/mmskeleton/tree/master/deprecated/origin_stgcn_repo).

The sequences are of the following dimension: TxVxC where:
 T is the number of time frames, V is the number of vertices and C is the number of features. The number of people in the time frame can be 1 or 2. Traditionally, for each vertex, there are three features (x,y,z). So if 2 people are there in a frame, then each vertex, there would be 2 sets of features: (x,y,z) for the first person and (x',y',z') for the second person. 
 So, for this experiment, the features of both people in a two person frame are padded along the 'C' dimension. So the feature per vertex per time frame would be (x,y,z,x',y',z') i.e 'C' would be 6. For a single person frame, the features of the second person would be filled with zeros i.e (x,y,z,0,0,0) indicating that there is no second person in the frame.

The sequences are batched along the 'T' dimension and this is taken care by the [DataListLoader](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.DataListLoader) of Pytorch Geometric. The number of time frames vary for each data. 

The *X_sub* protocol is followed. So the following are the training, validation and testing ids:
+ Train ids: 1,2,4,5,8,9,13,15,16,19,25,28,31,34,38
+ Validation ids: 14,17,18,27,35
+ Test ids: 3,6,7,10,11,12,20,21,22,23,24,26,29,30,32,33,36,37,39,40

The following hyper parameters are used:

 Optimizer  | Learning Rate |  Momentum    |  Batch Size | Epochs
 ------------- | -------------| ---------- | ---------| -------
 SGD           | 0.1          | 0.9        | 64   | 80 


+ The non-linear function used is ReLu. 
+ Dropout value used is 0.5. 
+ Number of STGCN Blocks used is 10.
+ Gnode used: gnode34
+ Number of GPUs used: 4
+ Edge importance parameter is set to True.

After 80 epochs, the training and validation plot: https://colab.research.google.com/drive/1xGrQRJGGHvwszpKDP-wTAO25ZrUnwMnJ?usp=sharing