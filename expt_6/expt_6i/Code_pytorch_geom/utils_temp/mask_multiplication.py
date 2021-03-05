import torch
import torch.nn as nn
import math  

def auto_padding(data,size):

	data_new = torch.zeros(size)
	lent = data.size(0)
	data_new[0:lent] = data 

	return data_new


class mask_multiplication(nn.Module):
	def __init__(self,kernel):
		super().__init__()

		self.kernel = kernel
		self.mask = nn.Parameter(torch.Tensor(1,self.kernel))
		self.bias = nn.Parameter(torch.Tensor(1,self.kernel))

		self.reset_parameters()


	def reset_parameters(self):
		nn.init.xavier_normal_(self.mask)
		nn.init.zeros_(self.bias)


	def forward(self,data):

		actual_size = data.size(0)

		# print("Actual size",actual_size)

		required_size = math.ceil(data.size(0)/self.kernel) * self.kernel

		# print("Required size",required_size)

		data_new = auto_padding(data,required_size).cuda()
		weight = self.mask 
		bias = self.bias
		weight = weight.repeat(1,(data_new.size(0)//self.kernel)).cuda()
		bias = bias.repeat(1,(data_new.size(0)//self.kernel)).cuda()
		data_new = data_new * weight[0] + bias[0]
		# print("New data",data_new)

		# for i in range((data_new.size(0)//self.kernel)):

		# 	data_new[self.kernel*i:self.kernel*(i+1)] = data_new[self.kernel*i:self.kernel*(i+1)] * self.mask

		data = data_new[0:actual_size]

		return data 



# mm = mask_multiplication(9)

# data = torch.ones(11)

# new_data = mm(data)

# print("New modified data",new_data)

