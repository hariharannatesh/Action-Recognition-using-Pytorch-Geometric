Date of Experiment: 31 - 12 - 2020

The following hyper parameters are used:

 Optimizer  | Learning Rate |  Momentum    |  Batch Size | Epochs
 ------------- | -------------| ---------- | ---------| -------
 SGD           | 0.05          | 0.9        | 32   | 120 


+ The non-linear function used is ReLu. 
+ Dropout value used is 0.5. 
+ Number of STGCN Blocks used is 10.
+ Gnode used: gnode16
+ Number of GPUs used: 2
+ Edge importance parameter is set to True.
+ Learning Rate Scheduler: scheduler=lr_scheduler.MultiStepLR(optimizer,milestones=[20,60],gamma=0.4)

After 120 epochs, the training and validation plot: https://colab.research.google.com/drive/1kV30E72pD4OnHJFJtKdn88DNbxiiSOMG#scrollTo=8fgQaR6v6C6x

Comparison of performance with sequence length and per class performance: https://colab.research.google.com/drive/1gpsrKWNo713CzlNaHDTGup0AEO12AU6n#scrollTo=4is2ONk9nMgL

Comparison of original STGCN Code with sequence length and per class performance: https://colab.research.google.com/drive/1FeLI59EJIRzlQcyfxMr02ux82R3cnoZ-#scrollTo=4is2ONk9nMgL

**Comparing Original STGCN Code and Pytorch Geometric Code Based on length**
https://docs.google.com/spreadsheets/d/10LksaQwa-0_j1uYCPTXebPuYeKOhC28MMu1-TeK8-Ts/edit#gid=0

**Comparing Original STGCN Code and Pytorch Geometric Code Based on Action Class**
https://docs.google.com/spreadsheets/d/11Weg7pmWkwoiSgAQ9spWzgbMYhKedH6iZaMB-2ecOW0/edit#gid=0

**Histogram of Validation data sequence length comparison and Training Data comparison (bin size 1)**
https://colab.research.google.com/drive/1xMblvIRHbCC-Bw983XgkYD_wDom7SuzF#scrollTo=DbGc8_iRNsu5
