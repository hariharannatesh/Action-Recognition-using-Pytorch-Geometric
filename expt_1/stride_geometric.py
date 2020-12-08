import torch
from torch_geometric.data import Batch
from torch_geometric.data import Data
import torch.nn as nn

class Sample_data():

	def forward(self,data,kernel,stride):

		x=data
		x=x.permute(0,3,1,2).contiguous()
		m,c,t,v=x.size()

		unfold=nn.Unfold(kernel_size=(1,v),stride=(stride,1))
		x_new=unfold(x)


		m,cv,t_new=x_new.size()

		x_new=x_new.view(m,c,v,t_new)

		x_new=x_new.permute(0,3,2,1).contiguous()


		data=x_new

		return data
