import torch
import os
import argparse
import torch_geometric
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Batch
import numpy as np 


def graph_create(data,action,kernel): #for kernel size of 3

	people,t,v,c=data.size()

	edge_index=[]
	before=0
	after=0
	p=(kernel-1)//2
	for i in range(t):
		# edge_index_new.append([i,i+1])
		# edge_index_new.append([i+1,i])
		if(i<p):
			num_zeros_before=p-i
			elems_exclude_cent=kernel-1-num_zeros_before
			before=i
			after=elems_exclude_cent-before
			indb=before
			inda=after

			while(indb!=0):
				edge_index.append([i,i-indb])
				indb=indb-1

			while(inda!=0):
				edge_index.append([i,i+inda])
				inda=inda-1
				
		elif(t-i<=p):
			num_zeros_after=p-(t-i)+1
			elems_exclude_cent=kernel-1- num_zeros_after
			before=kernel-1-p
			after = elems_exclude_cent - before
			indb=before 
			inda=after 

			while(indb!=0):
				edge_index.append([i,i-indb])
				indb=indb-1

			while(inda!=0):
				edge_index.append([i,i+inda])
				inda=inda-1

		else:
			for j in range(1,p+1):
				edge_index.append([i,i-j])
				edge_index.append([i,i+j])

	edge_index_np=np.asarray(edge_index).T
	edge_index_torch=torch.from_numpy(edge_index_np)

	
	if(people==2):
		data_1 = Data(x=data[0])
		data_2 = Data(x=data[1])
		data_list = [data_1,data_2]
		min_batch = Batch.from_data_list(data_list)
		# print("Mini batch",min_batch)
		data_graph = Data(x=min_batch.x,edge_index=edge_index_torch,y=action,m=people)

	else:
		data_graph = Data(x=data[0],edge_index=edge_index_torch,y=action,m=people)
	# print("New edge index",edge_index_new)
	return data_graph




class stgcn_dataset(Dataset):

	def __init__(self,directory):

		self.dir = directory
		self.files = os.listdir(self.dir)

	def __len__(self):
		return len(self.files)

	def __getitem__(self,idx):

		f = self.files[idx]
		data_np = np.load(os.path.join(self.dir,f))
		data_torch = torch.from_numpy(data_np)
		# data_torch = data_torch.permute(3,1,2,0).contiguous()

		action = int(f.split('.')[0].split('A')[1])
		data_graph = graph_create(data_torch,action,kernel=9)

		return data_graph
