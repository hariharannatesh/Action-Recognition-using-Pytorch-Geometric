Date of Experiment: 24 - 12 - 2020

The following hyper parameters are used:

 Optimizer  | Learning Rate |  Momentum    |  Batch Size | Epochs
 ------------- | -------------| ---------- | ---------| -------
 SGD           | 0.01          | 0.9        | 64   | 100 


+ The non-linear function used is ReLu. 
+ Dropout value used is 0.5. 
+ Number of STGCN Blocks used is 10.
+ Gnode used: gnode50
+ Number of GPUs used: 4
+ Edge importance parameter is set to True.
+ Learning Rate Scheduler: scheduler=lr_scheduler.MultiStepLR(optimizer,milestones=[10,50],gamma=0.5)

After 100 epochs, the training and validation plot: https://colab.research.google.com/drive/14WGA2fvpoXnGen4l_0XCGLlOD1Rki6FA#scrollTo=8fgQaR6v6C6x