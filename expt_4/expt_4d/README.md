Date of Experiment: 21 - 12 - 2020

The following hyper parameters are used:

 Optimizer  | Learning Rate |  Momentum    |  Batch Size | Epochs
 ------------- | -------------| ---------- | ---------| -------
 SGD           | 0.05          | 0.9        | 30   | 120 


+ The non-linear function used is ReLu. 
+ Dropout value used is 0.5. 
+ Number of STGCN Blocks used is 10.
+ Gnode used: gnode34
+ Number of GPUs used: 2
+ Edge importance parameter is set to True.
+ Learning Rate Scheduler: scheduler=lr_scheduler.MultiStepLR(optimizer,milestones=[75,100],gamma=0.96)

After 120 epochs, the training and validation plot: https://colab.research.google.com/drive/1mhYJlCMBUwc3nfVGxWX9X56bmSzvk449#scrollTo=8fgQaR6v6C6x