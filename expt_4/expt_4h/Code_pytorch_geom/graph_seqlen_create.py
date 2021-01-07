import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import Batch
import os 

# data=np.load('test_data_20.npy')
path_data='/ssd_scratch/cvit/nateshhariharan/Dataset_NTU_RGBD/ntu60_numpy/'

path_train='/ssd_scratch/cvit/nateshhariharan/Dataset_NTU_RGBD/train_ntu60_graph/'
path_test='/ssd_scratch/cvit/nateshhariharan/Dataset_NTU_RGBD/test_ntu60_graph/'

train_ids=["001","002","004","005","008","009","013","014","015","016","017","018","019","025","027","028","031","034","035","038"]
files=os.listdir(path_data)
# action_list=np.load('/ssd_scratch/cvit/nateshhariharan/nturgbd_skeletons_s001_to_s017/action_list.npy')
k=9
p=(k-1)//2
i=0
for f in files:

	data=np.load(os.path.join(path_data,f))

	lenth=data.shape[1]
	data_tensor=torch.from_numpy(data)
	action=int(f.split('A')[1].split('.')[0])
	print("Action dtype",type(action), action)
	# data_tensor=data_tensor.permute(1,0,2).contiguous()
	print(data_tensor.size())
	people,t,v,c=data_tensor.size()
	print("Number of people",people)
	edge_list=[]
	
	before=0
	after=0

	for j in range(t):

		if(j<p):
			num_zeros_before=p-j
			total_elements_exclude_cent=k-1-num_zeros_before
			before = j
			after = p
			indb=before
			inda= after

			while(indb!=0):
				edge_list.append([j,j-indb])
				indb=indb-1

			while(inda!=0):
				edge_list.append([j,j+inda])
				inda=inda-1



		elif(t-j<=p):
			num_zeros_after=p-(t-j)+1
			total_elements_exclude_cent = k-1-num_zeros_after
			before= p
			after= total_elements_exclude_cent - before
			indb=before
			inda=after 

			while(indb!=0):
				edge_list.append([j,j-indb])
				indb=indb-1

			while(inda!=0):
				edge_list.append([j,j+inda])
				inda=inda-1



		else:

			for m in range(1,p+1):
				edge_list.append([j,j-m])
				edge_list.append([j,j+m])


	edge_np=np.asarray(edge_list).T 
	edge_index=torch.from_numpy(edge_np).long().contiguous()
	# print(edge_index)
	if(people==2):
		data_1 = Data(x=data_tensor[0],edge_index=edge_index)
		data_2 = Data(x=data_tensor[1],edge_index=edge_index)
		data_list = [data_1,data_2]
		min_batch = Batch.from_data_list(data_list)
		print("Mini batch",min_batch)
		data_graph = Data(x=min_batch.x,edge_index=min_batch.edge_index,y=action,seq_len=t)

	else:
		data_graph = Data(x=data_tensor[0],edge_index=edge_index,y=action,seq_len=t)

	# print(data_graph)
	if(f.split('P')[1].split('R')[0] in train_ids):
		torch.save(data_graph,path_train+f.split('.')[0]+'.pt')
	else:
		torch.save(data_graph,path_test+f.split('.')[0]+'.pt')
	i=i+1


# datagraph=torch.load('./Dataset_full_len/test_data_20_0_graph.pt')
# print(datagraph.edge_index,datagraph)

