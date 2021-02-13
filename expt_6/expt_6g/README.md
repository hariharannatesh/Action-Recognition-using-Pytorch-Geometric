Date of Experiment: 12 - 2 - 2021

In this experiment, the sequences used are not padded and are of variable length.
The number of persons per data is 2 (i.e for a single person case, the sequences of one person is repeated again)
The GCNConv package from Pytorch Geometric is used for temporal convolution. A 1d convolution is performed on the edge_weight parameter to ensure that different weights are applied to different time nodes.


The following hyper parameters are used:

 Optimizer  | Learning Rate |  Momentum    |  Batch Size | Epochs
 ------------- | -------------| ---------- | ---------| -------
 SGD           | 0.1          | 0.9        | 64   | 80 


+ The non-linear function used is ReLu. 
+ Dropout value used is 0. 
+ Number of STGCN Blocks used is 10.
+ Gnode used: gnode21
+ Number of GPUs used: 4
+ Edge importance parameter is set to True.
+ Learning Rate Scheduler: scheduler=lr_scheduler.MultiStepLR(optimizer,milestones=[10,50],gamma=0.1)

After 80 epochs, the training and validation plot: https://colab.research.google.com/drive/1k2-d5iLYzvDGcRU0ol2RHbHaK0MySOou#scrollTo=8fgQaR6v6C6x
