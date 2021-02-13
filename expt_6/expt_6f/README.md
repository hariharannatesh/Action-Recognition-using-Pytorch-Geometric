Date of Experiment: 10 - 2 - 2021

In this experiment, the sequences used are not padded and are of variable length.
The number of persons per data is 2 (i.e for a single person case, the sequences of one person is repeated again)
nn.Conv1d is used for convolution instead of the GCNConv package from Pytorch Geometric. 


The following hyper parameters are used:

 Optimizer  | Learning Rate |  Momentum    |  Batch Size | Epochs
 ------------- | -------------| ---------- | ---------| -------
 SGD           | 0.1          | 0.9        | 40   | 80 


+ The non-linear function used is ReLu. 
+ Dropout value used is 0. 
+ Number of STGCN Blocks used is 10.
+ Gnode used: gnode39
+ Number of GPUs used: 2
+ Edge importance parameter is set to True.
+ Learning Rate Scheduler: scheduler=lr_scheduler.MultiStepLR(optimizer,milestones=[10,50],gamma=0.1)

After 80 epochs, the training and validation plot: https://colab.research.google.com/drive/1OOgEva-kI290p_kte4Tw0YOpqzMXTVBc#scrollTo=8fgQaR6v6C6x