import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.data import NeighborSampler
from torch_geometric.data import GraphSAINTSampler
import torch_geometric.utils as utils
import os

def change_edge_index(x,kernel): #for kernel size of 3

	edge_index_new=[]
	before=0
	after=0
	p=(kernel-1)//2
	t=len(x)
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
				edge_index_new.append([i,i-indb])
				indb=indb-1

			while(inda!=0):
				edge_index_new.append([i,i+inda])
				inda=inda-1
				
		elif(t-i<=p):
			num_zeros_after=p-(t-i)+1
			elems_exclude_cent=kernel-1- num_zeros_after
			before=kernel-1-p
			after = elems_exclude_cent - before
			indb=before 
			inda=after 

			while(indb!=0):
				edge_index_new.append([i,i-indb])
				indb=indb-1

			while(inda!=0):
				edge_index_new.append([i,i+inda])
				inda=inda-1

		else:
			for j in range(1,p+1):
				edge_index_new.append([i,i-j])
				edge_index_new.append([i,i+j])

	edge_index_new_np=np.asarray(edge_index_new).T
	edge_index_new_torch=torch.from_numpy(edge_index_new_np)
	# print("New edge index",edge_index_new)
	return edge_index_new_torch


class Sample_data():

	def forward(self,data,kernel,stride):

	    x=data
	    x=x.permute(2,0,1).contiguous()
	    c,t,v=x.size()
	    x=x.view(1,c,t,v)
	    # print("Original X",x)
	    unfold=nn.Unfold(kernel_size=(1,v),stride=(stride,1))
	    x_new=unfold(x)
	    x_new=x_new.permute(0,2,1).contiguous()
	    a,t_h,cv=x_new.size()
	    x_new=x_new.view(1,t_h,-1,v)
	    x_new=x_new.permute(0,1,3,2)
	    x_new=x_new.view(t_h,v,c)
	    # print("New x",x_new)
	    # print("Size",x_new.size())
	    # edge_index_new=change_edge_index(x_new,kernel)
	    # print("New edge index",edge_index_new)

	    return x_new

class Sample_batch(Sample_data):

	def forward(self,batch,kernel,stride):

		listt=Batch.to_data_list(batch)

		# print("Stride",stride)

		for i,data_graph in enumerate(listt):

			people = data_graph['m']

			if(people==1):

				x = data_graph.x 
				x,edge_index = Sample_data.forward(self,x,kernel,stride)

				data_graph.x = x 
				data_graph.edge_index = edge_index
				listt[i] = data_graph

			elif(people==2):

				x = data_graph.x 
				t = x.size(0)
				# print("T",t)
				x1 = x[0:t//2,:,:]
				x2 = x[t//2:t,:,:]

				x1 = Sample_data.forward(self,x1,kernel,stride)
				x2 = Sample_data.forward(self,x2,kernel,stride)

				data_1 = Data(x=x1)
				data_2 = Data(x=x2)
				batch_list = [data_1,data_2]
				mini_batch = Batch.from_data_list(batch_list)

				# print("Edge_index new",mini_batch.edge_index)

				data_graph.x = mini_batch.x
				listt[i] = data_graph





		batch_new=Batch.from_data_list(listt)

		# print("Batch vector",batch_new.batch)

		# print("Batch edge index",batch_new.edge_index)

		return batch_new
	  







	



