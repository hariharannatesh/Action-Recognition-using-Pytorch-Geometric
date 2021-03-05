import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from torch_geometric.data import DataListLoader
from torch_geometric.data import Data 
from torch_geometric.data import Dataset
from torch_geometric.data import Batch

import os  
import torch_geometric.utils as utils
import torch.optim as optim
from torch_geometric.nn import global_mean_pool as GMP 
from torch_geometric.nn import BatchNorm as graph_bn
from torch_sparse import SparseTensor

from .stride_geometric import Sample_batch
from .graph_2 import Graph
from .pyg_batchnorm import BatchNorm1d,BatchNorm2d
from .mask_multiplication import mask_multiplication

import argparse

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import DataParallel
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
# torch.set_printoptions(precision=10)
class Net_time(nn.Module):
	def __init__(self,
				input_channels,
				output_channels,
				edge_importance_weighting,
				kernel,
				stride,
				residual):
		super().__init__()
		self.inc=input_channels
		self.outc=output_channels
	
		self.k = adj_old.size(0)
		self.stride=stride 
		self.residual = residual
		self.kernel = kernel 

		self.nnode=Net_nodes(input_channels,output_channels,self.k)


		self.tcnconv = GCNConv(output_channels,output_channels,normalize=False)
		# self.tcnconv.weight = nn.Parameter(torch.ones_like(self.tcnconv.weight))
		# self.tcnconv.bias = nn.Parameter(torch.ones_like(self.tcnconv.bias))

		self.mm = mask_multiplication(self.kernel)

		self.bn_2d1 = BatchNorm2d(output_channels)
		self.bn_2d2 = BatchNorm2d(output_channels)
		self.bn_2d3 = BatchNorm2d(output_channels)
		self.s=Sample_batch()



		if(edge_importance_weighting==True):
			self.edge_importance=nn.Parameter(torch.ones(adj_old.size()))
		else:
			self.edge_importance=torch.tensor(1)

		if(residual):
			self.res = nn.Conv1d(input_channels,output_channels,stride=1,kernel_size=1)
			# self.res.weight = nn.Parameter(torch.ones_like(self.res.weight))
			# self.res.bias = nn.Parameter(torch.ones_like(self.res.bias))
		# print("Time conv weights",self.conv1.weight)

	def forward(self,data):

		x,edge_index_t=data.x,data.edge_index

		adj_old_new=adj_old.to(x.device)

		A=adj_old_new * self.edge_importance

		x_res=torch.tensor(0)
		# x_res=x_res.to(device)

		if(self.residual):
			if(self.inc==self.outc):
				x_res=x
	

			else:
				
				x_res=x
				x_res=x_res.type(torch.float32)
				x_res = x_res.permute(0,2,1).contiguous()
				x_res = self.res(x_res)
				x_res = x_res.permute(0,2,1).contiguous()
				# x_res=torch.matmul(x_res.float(),self.res_weights.float()) + self.res_bias.float() #remember to add bias
				data_res=data
				data_res.x=x_res
				if(self.stride!=1):
					data_res=self.s.forward(data_res,self.kernel,self.stride) 
				x_res=self.bn_2d1(data_res.x)
				# x_res = data_res.x
				


		else:
			x_res=torch.tensor(0)


		x=self.nnode(x=x,edge_index=A)
		
		data.x = x 

		data.x = self.bn_2d2(data.x)

		data.x=F.relu(data.x)

		edge_weight=torch.ones((edge_index_t.size(1), ),device=x.device)
		# print("Edge weight size",edge_weight.size())
		edge_index_t = edge_index_t.cuda()

		edge_index_t,edge_weight =utils.add_self_loops(edge_index_t,edge_weight)

		# lent = edge_weight.size(0)

		# edge_weight=edge_weight.view(1,1,lent)
		# convolved_edge_weight = self.edge_conv(edge_weight)
		# convolved_edge_weight = convolved_edge_weight.view(self.kernel,lent)

		# data.x = torch.repeat_interleave(data.x,repeats=self.kernel,dim=0)
		# data.x = data.x.repeat(self.kernel,1,1)
		# print("Old edge weight",edge_weight)
		edge_weight = self.mm(edge_weight)

		# print("New edge weight",edge_weight)


		data.x=data.x.permute(1,0,2).contiguous()
		# x1 = data.x 
		# x2 = torch.zeros_like(data.x)
		data.x=self.tcnconv(data.x,edge_index_t,edge_weight=edge_weight)

		data.x=data.x.permute(1,0,2).contiguous()



		if(self.stride!=1):
			data=self.s.forward(data,self.kernel,self.stride)	

		data.x=self.bn_2d3(data.x.float())
		data.x = F.dropout(data.x,p=0,training=self.training,inplace=True)
		
		data.x=data.x.type(torch.float32)
		data.x = data.x + x_res
		data.x=F.relu(data.x)

		return data



class Net_nodes(nn.Module):
	
	def __init__(self,
				input_channels,
				output_channels,
				kernel_size):
		super().__init__()

		self.k = kernel_size
		self.spconv = nn.Conv1d(input_channels,self.k*output_channels,kernel_size=1,stride=1)
		# self.spconv.weight = nn.Parameter(torch.ones_like(self.spconv.weight))
		# self.spconv.bias = nn.Parameter(torch.ones_like(self.spconv.bias))
		


	def forward(self,x,edge_index):
		# print("X type",x.dtype)
		# print("Edge index dtype",edge_index.dtype)
		x = x.permute(0,2,1).contiguous()
		x_sum = self.spconv(x)
		t,kc,v = x_sum.size()
		x_sum = x_sum.view(t,self.k,kc//self.k,v)
		x_sum = torch.einsum('tkcv,kvw->tcw',(x_sum,edge_index))
		x_sum = x_sum.permute(0,2,1).contiguous()
		x_sum = x_sum.to(x.device)

		
		return x_sum


class Model(nn.Module):

	def __init__(self,input_channels,output_channels,num_classes):

		super().__init__()

		self.bn_1d=BatchNorm1d(25*input_channels)

		self.stgcn_networks = nn.ModuleList((
			Net_time(input_channels,64,edge_importance_weighting=True,kernel=9,stride=1,residual=False),
			Net_time(64,64,edge_importance_weighting=True,kernel=9,stride=1,residual=True),
			Net_time(64,64,edge_importance_weighting=True,kernel=9,stride=1,residual=True),
			Net_time(64,64,edge_importance_weighting=True,kernel=9,stride=1,residual=True),
			Net_time(64,128,edge_importance_weighting=True,kernel=9,stride=2,residual=True),
			Net_time(128,128,edge_importance_weighting=True,kernel=9,stride=1,residual=True), #commented for now
			Net_time(128,128,edge_importance_weighting=True,kernel=9,stride=1,residual=True),
			Net_time(128,256,edge_importance_weighting=True,kernel=9,stride=2,residual=True),
			Net_time(256,256,edge_importance_weighting=True,kernel=9,stride=1,residual=True),
			Net_time(256,output_channels,edge_importance_weighting=True,kernel=9,stride=1,residual=True),
			))




		self.prediction = nn.Conv1d(output_channels,num_classes,kernel_size=1,stride=1)
		# self.prediction.weight = nn.Parameter(torch.ones_like(self.prediction.weight))
		# self.prediction.bias = nn.Parameter(torch.ones_like(self.prediction.bias))
		# nn.init.zeros_(self.prediction_bias)


	def forward(self,data):

		# print("Data before conv inside module ",data.num_graphs,data.batch.device,data.x.size())
		# print("Data y",data.y)

		data.x=self.bn_1d(data.x.float())
		op = data 

		for stgcn in self.stgcn_networks:

			op = stgcn(op)

		# x_result=op.x.type(torch.float64)
		x_result = op.x

		op.batch=op.batch.cuda()
		# print("X_result device",x_result.device)

		# print("Batch device",op.batch.device)

		x_avg=GMP(x_result,batch=op.batch)

		x_mean=x_avg.mean(dim=1)


		# print("Avg",x_mean)
		batch,channels = x_mean.size()
		x_mean = x_mean.view(batch,channels,1)
		predicted_val = self.prediction(x_mean)
		predicted_val = predicted_val.view(batch,-1)



		return predicted_val



gr=Graph()
adj_old=torch.tensor(gr.A,dtype=torch.float32,requires_grad=False)
