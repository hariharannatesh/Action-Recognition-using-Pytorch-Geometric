Date of Experiment: 12 - 1 - 2021

In this experiment, the data is padded upto 300 frame sequences. 
The number of persons per data is 2 (i.e for a single person case, the sequences of one person is repeated again)

The following hyper parameters are used:

 Optimizer  | Learning Rate |  Momentum    |  Batch Size | Epochs
 ------------- | -------------| ---------- | ---------| -------
 SGD           | 0.05          | 0.9        | 20   | 100 


+ The non-linear function used is ReLu. 
+ Dropout value used is 0.5. 
+ Number of STGCN Blocks used is 10.
+ Gnode used: gnode54
+ Number of GPUs used: 4
+ Edge importance parameter is set to True.
+ Learning Rate Scheduler: scheduler=lr_scheduler.MultiStepLR(optimizer,milestones=[20,60],gamma=0.4)

After 100 epochs, the training and validation plot: https://colab.research.google.com/drive/1GW5-TWUnkef6fGtmTt0_g5N1p3UFGtOC#scrollTo=8fgQaR6v6C6x
