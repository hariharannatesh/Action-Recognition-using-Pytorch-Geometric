import torch
import os
import argparse
import torch_geometric
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Batch
import numpy as np 


def graph_create(data,action): #for kernel size of 3

	people,t,v,c=data.size()

	
	if(people==2):
		data_1 = Data(x=data[0])
		data_2 = Data(x=data[1])
		data_list = [data_1,data_2]
		min_batch = Batch.from_data_list(data_list)
		# print("Mini batch",min_batch)
		data_graph = Data(x=min_batch.x,y=action,m=people)

	else:
		data_graph = Data(x=data[0],y=action,m=people)
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

		action = int(f.split('.')[0].split('A')[1])
		data_graph = graph_create(data_torch,action)

		return data_graph
