Date of Experiment: 8 - 2 - 2021

In this experiment, the data is padded upto 300 frame sequences. 
The number of persons per data is 2 (i.e for a single person case, the sequences of one person is repeated again)
nn.Conv1d is used for convolution instead of the GCNConv package from Pytorch Geometric. 

The following hyper parameters are used:

 Optimizer  | Learning Rate |  Momentum    |  Batch Size | Epochs
 ------------- | -------------| ---------- | ---------| -------
 SGD           | 0.1          | 0.9        | 30   | 51 


+ The non-linear function used is ReLu. 
+ Dropout value used is 0.5. 
+ Number of STGCN Blocks used is 10.
+ Gnode used: gnode59
+ Number of GPUs used: 4
+ Edge importance parameter is set to True.
+ Learning Rate Scheduler: scheduler=lr_scheduler.MultiStepLR(optimizer,milestones=[10,50],gamma=0.1)

After 51 epochs, the training plot: https://colab.research.google.com/drive/188ceuWIg-N9-xhtR4gUxL6bjlqA50JBt#scrollTo=8fgQaR6v6C6x

Number of test samples: 16478
Accuracy of Model : 0.72842
