import torch
import torch.nn as nn
class BatchNorm1d(nn.Module):

	def __init__(self,channels):
		super().__init__()

		self.weight = nn.Parameter(torch.ones(channels,1))
		self.bias = nn.Parameter(torch.zeros(channels,1))

	def forward(self,x):
		T,V,C = x.size()
		x = x.permute(1,2,0).contiguous()
		x = x.view(C*V,T)


		mean = torch.mean(x,dim=1,keepdim=True)
		mean = mean.type(x.dtype)

		var = torch.var(x,dim=1,keepdim=True,unbiased=False)
		var = var.type(x.dtype)

		x_bn = ((x-mean)/torch.sqrt(var+ 1e-05)) * self.weight + self.bias
		x_bn = x_bn.type(x.dtype)

		x_bn = x_bn.view(V,C,T)

		x_bn = x_bn.permute(2,0,1).contiguous()

		return x_bn


class BatchNorm2d(nn.Module):

	def __init__(self,channels):
		super().__init__()

		self.weight = nn.Parameter(torch.ones(channels))
		self.bias = nn.Parameter(torch.zeros(channels))

	def forward(self,x):
		summ = torch.sum(x,dim=1,keepdim=True).sum(dim=0,keepdim=True)
		mean = summ/(x.size(0)*x.size(1))

		diff_with_mean = x - mean 

		sq_with_mean = torch.square(diff_with_mean)
		variance = torch.sum(sq_with_mean,dim=1,keepdim=True).sum(dim=0,keepdim=True)/((x.size(0)*x.size(1)))

		x_bn = (diff_with_mean/torch.sqrt(variance + 1e-05)) * self.weight + self.bias

		return x_bn 


