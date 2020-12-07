In this experiment, the code in Pytorch Geometric is as close to the original STGCN Code (https://github.com/open-mmlab/mmskeleton/tree/master/deprecated/origin_stgcn_repo).

The sequences are of the following dimension: MxTxVxC where:
M is the number of people, T is the number of time frames, V is the number of vertices and C is the number of features.

The sequence is batched along the 'M' dimension and all the sequences are padded to a length of 300 to ensure uniformity.

The *X_sub* protocol is followed. So the following are the training, validation and testing ids:
+ Train ids: 1,2,4,5,8,9,13,15,16,19,25,28,31,34,38
+ Validation ids: 14,17,18,27,35
+ Test ids: 3,6,7,10,11,12,20,21,22,23,24,26,29,30,32,33,36,37,39,40

The following hyper parameters are used:

Optimizer | Learning Rate | Weight Decay | Momentum | Batch Size | Epochs
_ _ _ _ _  |_ _ _ _ _ _ _ | _ _ _ _ _ _  |_ _ _ _ _ | _ _ _ _ _ _ | _ _ _

SGD        | 0.1          | 0.001        | 0.9 (Nesterov=True)| 40 | 80

The non-linear function used is ReLu. Dropout value used is 0.5. 