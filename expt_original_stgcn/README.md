The original stgcn code is used to train the model from scratch to observe the plots. (https://github.com/open-mmlab/mmskeleton/tree/master/deprecated/origin_stgcn_repo).


The *X_sub* protocol is followed. So the following are the training, validation and testing ids:
+ Train ids: 1,2,4,5,8,9,13,15,16,19,25,28,31,34,38
+ Validation ids: 14,17,18,27,35
+ Test ids: 3,6,7,10,11,12,20,21,22,23,24,26,29,30,32,33,36,37,39,40

The following hyper parameters are used:

 Optimizer  | Learning Rate |  Momentum |  Batch Size | Epochs
 ------------- | -------------| ---------- | ---------| -------
 SGD           | 0.1          | 0.9 (Nesterov=True)| 40 | 80 

Date: 18/12/2020

+ The non-linear function used is ReLu. 
+ Dropout value used is 0.5. 
+ Number of STGCN Blocks used is 10.
+ Gnode used: gnode11
+ Edge importance parameter is set to True
+ Learning Rate Scheduler: scheduler=lr_scheduler.MultiStepLR(optimizer,milestones=[10,50],gamma=0.1)

Link for the Train and Validation Plots: https://colab.research.google.com/drive/1bXkdLHb_a4gmUgfw161nUSNWq8hUZRPo#scrollTo=8fgQaR6v6C6x
