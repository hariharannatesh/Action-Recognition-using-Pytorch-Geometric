Date of Experiment: 28 - 12 - 2020

The following hyper parameters are used:

 Optimizer  | Learning Rate |  Momentum    |  Batch Size | Epochs
 ------------- | -------------| ---------- | ---------| -------
 SGD           | 0.01          | 0.9        | 30   | 100 


+ The non-linear function used is ReLu. 
+ Dropout value used is 0.5. 
+ Number of STGCN Blocks used is 5.
+ Gnode used: gnode38
+ Number of GPUs used: 2
+ Edge importance parameter is set to True.
+ Learning Rate Scheduler: scheduler=lr_scheduler.MultiStepLR(optimizer,milestones=[10,50],gamma=0.1)

After 100 epochs, the training and validation plot: 