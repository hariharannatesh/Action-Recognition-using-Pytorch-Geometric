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

Range of sequence length | Number of validation samples | Correctly recognized by torch geometric code |Percentage
-------------------------|------------------------------|----------------------------------------------|----------
 0-50                    | 574                          | 382                                          | 0.6655
 50-100                  | 5090                         | 3080                                         | 0.605
 100-150                 | 2579                         | 1591                                         | 0.616
 150-200                 | 839                          | 542                                          | 0.646
 200-250                 | 155                          | 91                                           | 0.587
 250-300                 | 36                           | 19                                           | 0.5277
 300-350                 | 14                           | 0                                            | 0.0
 350-400                 | 2                            | 1                                            | 0.5
 400-450                 | 0                            | 0                                            | -
 450-500                 | 0                            | 0                                            | -
 500-550                 | 1                            | 0                                            | 0


Per class performance:

Class Name    | Number of samples | Correctly recognized by original stgcn(A) | Correctly recognized by pytorch geometric(B) |  B-A   
--------------|-------------------|-------------------------------------------|----------------------------------------------|--------
1. Drink water| 155               |                                           | 72                                           |
2. Eat meal   | 154               |                                           | 60                                           |
3. Brushing teeth|154             |                                           | 83                                           |        
4.Brushing hair  |155             |                                           | 52                                           |        
5.Drop           |153             |                                           | 85                                           |
6.Pickup         |155             |                                           | 141                                          |
7.Throw          |154             |                                           | 101                                          |
8.Sit down       |155             |                                           | 123                                          |
9.Stand up(from sit)|153          |                                           | 129                                          |
10.Clapping      | 153            |                                           | 45                                           |
11.Reading       | 154            |                                           | 41                                           |
12.Writing       | 154            |                                           | 69                                           |
13.Tear paper    | 152            |                                           | 100                                          |
14.Wear jacket   | 155            |                                           | 148                                          |
15.Take off jacket|155            |                                           | 116                                          |
16.Wear a shoe    | 154           |                                           | 73                                           |
17.Take off shoe | 155            |                                           | 78                                           |
18.Wear glasses  | 154            |                                           | 91                                           |
19.Take glasses  | 154            |                                           | 73                                           |
20.Put hat/cap   | 154            |                                           | 116                                          |
21.Take hat/cap  | 154            |                                           | 112                                          |
22.Cheer up      | 156            |                                           | 99                                           |
23.Hand waiving  | 156            |                                           | 61                                           |
24.Kicking something|155          |                                           | 116                                          |
25.Reach into pocket|156          |                                           | 66                                           |
26.Hopping          |156          |                                           | 145                                          |
27.Jump up          |156          |                                           | 151                                          |
28.make phone/answer|154          |                                           | 47                                           |
29.Play with phone  |154          |                                           | 65                                           |
30.Typing on keyboard|155         |                                           | 44                                           |
31.Pointing with finger|156       |                                           | 76                                           |
32.Taking selfie       |156       |                                           | 68                                           |
33.Check time (watch)|156         |                                           | 94                                           |
34.Rub two hands|156              |                                           |84                                            |
35.Nod head/bow |156              |                                           |123                                           |
36.Shake hand   |156              |                                           |117                                           |
37.Wipe face    |155              |                                           |76                                            |
38.Salute       |156              |                                           |107                                           |
39.Put palms together|156         |                                           |86                                            |
40.Cross hands infront|153        |                                           |119                                           |
41.Sneeze/Cough|156               |                                           |77                                            |
42.Staggering  |156               |                                           |109                                           |
43.Falling     |156               |                                           |145                                           |
44.Touch head(headache)|156       |                                           |59                                            |
45.Touch chest(stomachache)|155   |                                           |57                                            |
46.Touch back (backache)|155      |                                           |60                                            |
47.Touch neck(neckache)|156       |                                           |45                                            |
48.Nausea      |156               |                                           |79                                            |
49.Use fan/feel warm|156          |                                           |57                                            |
50.Punching/slapping |154         |                                           |105                                           |
51.Kicking person    |156         |                                           |130                                           |
52.Pushing person    |156         |                                           |141                                           |
53.Pat other person  |156         |                                           |91                                            |
54.Point finger person|153        |                                           |133                                           |
55.Hugging            |145        |                                           |125                                           |
56.Giving to other    |156        |                                           |124                                           |
57.Touching other person pocket|155|                                          |119                                           |
58.Handshaking        |156        |                                           |133                                           |
59.Walking towards each other|155 |                                           |135                                           |
60.Walking away each other   |156 |                                           |135                                           |                                                                                                                                                 