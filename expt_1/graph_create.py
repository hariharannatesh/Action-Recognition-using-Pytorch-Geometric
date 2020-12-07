import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
import os 

# data=np.load('test_data_20.npy')
path_data='/ssd_scratch/cvit/nateshhariharan/Dataset_NTU_RGBD/ntu60_numpy_300padded/'

path_train='/ssd_scratch/cvit/nateshhariharan/Dataset_NTU_RGBD/train_ntu60padded_graph/'
path_test='/ssd_scratch/cvit/nateshhariharan/Dataset_NTU_RGBD/test_ntu60padded_graph/'

train_ids=["001","002","004","005","008","009","013","014","015","016","017","018","019","025","027","028","031","034","035","038"]
files=os.listdir(path_data)

for f in files:

	data=np.load(os.path.join(path_data,f))

	data_torch=torch.from_numpy(data)

	print("Size",data_torch.size())

	action=int(f.split('.')[0].split('A')[1])

	noperson=data_torch.size(0)

	edge_index=[]

	if(noperson==1):
		edge_index.append([0,0])

	elif(noperson==2):
		edge_index.append([0,1])
		edge_index.append([1,0])

	edge_index_np=np.asarray(edge_index).T 
	edge_index_torch=torch.from_numpy(edge_index_np)

	graph_obj=Data(x=data_torch,y=action,edge_index=edge_index_torch)

	print("Data object",graph_obj.edge_index)

	if(f.split('P')[1].split('R')[0] in train_ids):
		torch.save(graph_obj,path_train+f.split('.')[0]+'.pt')
	else:
		torch.save(graph_obj,path_test+f.split('.')[0]+'.pt')



# datagraph=torch.load('./Dataset_full_len/test_data_20_0_graph.pt')
# print(datagraph.edge_index,datagraph)

