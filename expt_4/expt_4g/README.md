Date of Experiment: 31 - 12 - 2020

The following hyper parameters are used:

 Optimizer  | Learning Rate |  Momentum    |  Batch Size | Epochs
 ------------- | -------------| ---------- | ---------| -------
 SGD           | 0.05          | 0.9        | 32   | 120 


+ The non-linear function used is ReLu. 
+ Dropout value used is 0.5. 
+ Number of STGCN Blocks used is 10.
+ Gnode used: gnode62
+ Number of GPUs used: 2
+ Edge importance parameter is set to True.
+ Learning Rate Scheduler: scheduler=lr_scheduler.MultiStepLR(optimizer,milestones=[20,60],gamma=0.8)

After 120 epochs, the training and validation plot: https://colab.research.google.com/drive/1yNEB7kbS_BgZ3ayLVauN0B0OINze8jQc#scrollTo=8fgQaR6v6C6x
