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
from torch.nn import BatchNorm1d,BatchNorm2d

import os  
import torch_geometric.utils as utils
import torch.optim as optim
from torch_geometric.nn import global_mean_pool as GMP 
from torch_sparse import SparseTensor

from stride_geometric import Sample_data
from graph_2 import Graph

import argparse

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import DataParallel
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
# torch.set_printoptions(precision=10)


def time_edge_index(time_len,kernel): 

	edge_index_new=[]
	before=0
	after=0
	t=time_len
	p=(kernel-1)//2
	for i in range(t):
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



class Net_time(nn.Module):
	def __init__(self,
				input_channels,
				output_channels,
				edge_importance_weighting):
		super().__init__()
		self.inc=input_channels
		self.outc=output_channels
		self.nnode=Net_nodes(input_channels,output_channels)
		self.conv1=GCNConv(output_channels,output_channels,node_dim=2,bias=True,normalize=False)
		nn.init.xavier_normal_(self.conv1.weight,gain=nn.init.calculate_gain('relu'))
		nn.init.zeros_(self.conv1.bias)
		self.conv1.weight=nn.Parameter(self.conv1.weight.type(torch.float64))
		
		self.bn_2d=BatchNorm2d(output_channels)
		self.drop=nn.Dropout(0.5,inplace=True)

		self.s=Sample_data()

		if(edge_importance_weighting==True):
			self.edge_importance=nn.Parameter(torch.ones(adj_old.size()))
		else:
			self.edge_importance=torch.tensor(1)

		

	def forward(self,data,stride=1,kernel=9,residual=False):

		x=data.x 

		edge_index_t = time_edge_index(x.size(1),kernel)

		data_final=data

		adj_old_new=adj_old.to(x.device)

		A=adj_old_new * self.edge_importance

		edge_index_sp=[]
		for i in range(len(A)):
			edge_index_sp.append(SparseTensor.from_dense(A[i]))


		x_res=torch.tensor(0)

		if(residual):
			if(self.inc==self.outc):
				x_res=x

			else:
				
				weights=nn.Parameter(torch.empty(self.inc,self.outc,dtype=torch.float32)).cuda()
				weight=nn.init.xavier_normal_(weights,gain=nn.init.calculate_gain('relu')).cuda()
				x_res=x
				x_res=x_res.type(torch.float32)
				x_res=torch.matmul(x_res,weights)

				if(stride!=1):
					x_res=self.s.forward(x_res,kernel,stride) 
				# print("Residual data device",data_res['x'].device)
				x_res=x_res.permute(0,3,1,2).contiguous()
				x_res=self.bn_2d(x_res.double())
				x_res=x_res.permute(0,2,3,1).contiguous()
	


		else:
			x_res=torch.tensor(0)
			
		edge_weight=torch.ones((edge_index_t.size(1), ),dtype=torch.long).cuda()

		

		x=self.nnode(x=x,edge_index=edge_index_sp)

		x=x.permute(0,3,1,2).contiguous()
		x=self.bn_2d(x)
		x=F.relu(x)
		x=x.permute(0,2,3,1).contiguous()
	
		edge_index_t,edge_weight=utils.add_self_loops(edge_index_t,edge_weight)
	

		edge_index_t=edge_index_t.cuda()

		x=x.permute(0,2,1,3).contiguous()
		x=self.conv1(x,edge_index_t,edge_weight=edge_weight)
		x=x.permute(0,2,1,3).contiguous()
		x=x.permute(0,3,1,2).contiguous()
		x=self.bn_2d(x)
		x=self.drop(x)
		x=x.permute(0,2,3,1).contiguous()


		data.x = x

		data_final['x']=x

		if(stride!=1):
			data_final.x=self.s.forward(data_final.x,kernel,stride)

		data_final.x = data_final.x + x_res
		
		data_final.x=F.relu(data_final.x)

		return data_final



class Net_nodes(nn.Module):
	
	def __init__(self,
				input_channels,
				output_channels):
		super().__init__()
		self.conv1=GCNConv(input_channels,output_channels,node_dim=2,bias=True,cached=True)
		# self.conv1.weight.detach().cpu()
		self.conv1.weight=nn.Parameter(torch.empty(self.conv1.weight.size(),dtype=torch.float64))
		nn.init.xavier_normal_(self.conv1.weight,gain=nn.init.calculate_gain('relu'))
		nn.init.zeros_(self.conv1.bias)
		# print("Self conv1 weight device inside class",self.conv1.weight)

		self.conv2=GCNConv(input_channels,output_channels,node_dim=2,bias=True,cached=True)
		self.conv2.weight=nn.Parameter(torch.empty(self.conv2.weight.size(),dtype=torch.float64))
		nn.init.xavier_normal_(self.conv2.weight,gain=nn.init.calculate_gain('relu'))
		nn.init.zeros_(self.conv2.bias)
		
		self.conv3=GCNConv(input_channels,output_channels,node_dim=2,bias=True,cached=True)
		self.conv3.weight=nn.Parameter(torch.empty(self.conv3.weight.size(),dtype=torch.float64))
		nn.init.xavier_normal_(self.conv3.weight,gain=nn.init.calculate_gain('relu'))
		nn.init.zeros_(self.conv3.bias)
		

	def forward(self,x,edge_index):
		
		x_sum=[]
		# print("Edge index device",edge_index[0].device)
		# self.conv1._cached_adj_t=edge_index[0].device_as(x,non_blocking=True)
		self.conv1._cached_adj_t=edge_index[0]
		self.conv1._cached_adj_t=self.conv1._cached_adj_t.type_as(x)
		sum1=self.conv1(x,edge_index[0])
		x_sum.append(sum1)


		self.conv2._cached_adj_t=edge_index[1]
		self.conv2._cached_adj_t=self.conv2._cached_adj_t.type_as(x)
		sum2=self.conv2(x,edge_index[1])
		x_sum.append(sum2)


		self.conv3._cached_adj_t=edge_index[2]
		self.conv3._cached_adj_t=self.conv3._cached_adj_t.type_as(x)
		sum3=self.conv3(x,edge_index[2])
		x_sum.append(sum3)

		x_sum=torch.stack(x_sum,dim=0).sum(dim=0).to(x.device)

		print("X_sum size",x_sum.size())

		return x_sum


class Model(nn.Module):

	def __init__(self,input_channels,output_channels,num_classes):

		super().__init__()

		self.bn_1d=BatchNorm1d(input_channels*25)
		self.stgcn1=Net_time(input_channels,64,edge_importance_weighting=True)
		self.stgcn2=Net_time(64,64,edge_importance_weighting=True)
		self.stgcn3=Net_time(64,64,edge_importance_weighting=True)
		self.stgcn4=Net_time(64,64,edge_importance_weighting=True)
		self.stgcn5=Net_time(64,128,edge_importance_weighting=True)
		self.stgcn6=Net_time(128,128,edge_importance_weighting=True)
		self.stgcn7=Net_time(128,128,edge_importance_weighting=True)
		self.stgcn8=Net_time(128,256,edge_importance_weighting=True)
		# # self.stgcn9=Net_time(256,output_channels) #newly added
		self.stgcn9=Net_time(256,256,edge_importance_weighting=True)
		self.stgcn10=Net_time(256,output_channels,edge_importance_weighting=True)
		
		self.fcn=nn.Conv2d(output_channels,num_classes,kernel_size=1)
		self.fcn.weight=nn.Parameter(torch.ones_like(self.fcn.weight))
		
		self.softmax=nn.Softmax(dim=1)


	def forward(self,data):

		# print("Data before conv inside module ",data.num_graphs,data.batch.device,data.x.size())
		# print("Data y",data.y)

		x=data.x.permute(0,2,3,1).contiguous()
		M,V,C,T=x.size()
		x=x.view(M,V*C,T)
		x=self.bn_1d(x)

		x=x.view(M,V,C,T).contiguous()
		x=x.permute(0,3,1,2).contiguous()
		data.x=x 

		op=self.stgcn1(data,kernel=9)
		op=self.stgcn2(op,kernel=9,residual=True)
		op.x=op.x.type(torch.float64)
		op=self.stgcn3(op,kernel=9,residual=True)
		op.x=op.x.type(torch.float64)
		op=self.stgcn4(op,kernel=9,residual=True)
		op.x=op.x.type(torch.float64)
		op=self.stgcn5(op,kernel=9,stride=2,residual=True)
		op.x=op.x.type(torch.float64)
		op=self.stgcn6(op,kernel=9,residual=True)
		op.x=op.x.type(torch.float64)
		op=self.stgcn7(op,kernel=9,residual=True)
		op.x=op.x.type(torch.float64)
		op=self.stgcn8(op,kernel=9,stride=2,residual=True)
		op.x=op.x.type(torch.float64)
		op=self.stgcn9(op,kernel=9,residual=True)
		op.x=op.x.type(torch.float64)
		op=self.stgcn10(op,kernel=9,residual=True)
		op.x=op.x.type(torch.float64)

		# print(op.x.size())
		x_result=op.x.type(torch.float64)
		x_result=x_result.permute(0,3,1,2).contiguous()

		x_avg=F.avg_pool2d(x_result,x_result.size()[2:])

		op.batch=op.batch.cuda()	

		x_mean=GMP(x_avg,batch=op.batch)

		# print("Avg",x_mean)

		predicted_val=self.fcn(x_mean)
		predicted_val=predicted_val.view(1,-1).contiguous()
		# print("Predicted val",predicted_val)


		return predicted_val

# path_time='./Dataset_full_len2/'

parser=argparse.ArgumentParser()

parser.add_argument("--path_train")
parser.add_argument("--path_val")
parser.add_argument("--epochs",type=int)
parser.add_argument("--batch_size",type=int)

args = parser.parse_args()
data_list_time=[]
# files_t=os.listdir(path_time)
files_t=os.listdir(args.path_train)
# files=files[0:10]
# files_t=files_t[0:10]
# print("Files_t",files_t)

files_v=os.listdir(args.path_val)
# files_t,files_v=train_test_split(files,test_size=0.2,random_state=42)
# files_v=files_v[0:4]
# print("Files v",files_v)
seq_len=[]



gr=Graph()
adj_old=torch.tensor(gr.A,dtype=torch.float32,requires_grad=False)


# print("Edge index sp",edge_index_sp)



# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loader={'train':DataListLoader([torch.load(os.path.join(args.path_train,f)) for f in files_t],batch_size=args.batch_size,shuffle=False),
		'val':DataListLoader([torch.load(os.path.join(args.path_val,f)) for f in files_v],batch_size=args.batch_size,shuffle=False)}

print("Loader train:",len(loader['train']),"Val train",len(loader['val']))

dataset_sizes={'train':len(files_t),'val':len(files_v)}
print("Dataset sizes",dataset_sizes)

model=Model(3,256,num_classes=60).double()
# model=Model(3,128,num_classes=60).double()
# model=Model(3,64,num_classes=60).double()
# print("Weights of spatial conv",getattr(model,"Spconv"))

if(torch.cuda.device_count()>1):
	print("Number of gpus",torch.cuda.device_count())
	model=DataParallel(model,device_ids=[0,1],output_device=torch.device('cuda').index)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# print("Device",device)

writer={'train':SummaryWriter('./runs/expt_val1/train'),'val':SummaryWriter('./runs/expt_val1/val')}
# epochs=1
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.1,momentum=0.9,nesterov=True,weight_decay=0.0001)
# scheduler=lr_scheduler.StepLR(optimizer,step_size=80,gamma=0.96)
# scheduler=lr_scheduler.MultiStepLR(optimizer,milestones=[8,50,100],gamma=0.96)
# print("Model parameters",list(model.named_parameters()))
train_loss=[]
train_accuracy=[]
val_loss=[]
val_accuracy=[]


for epoch in range(args.epochs):

	epoch_loss={'train':0.0,'val':0.0}
	total={'train':0.0,'val':0.0}
	# correct={'train':0.0,'val':0.0}
	accuracy={'train':0.0,'val':0.0}

	for phase in loader.keys():

		print("Phase",phase)

		if(phase=='train'):
			torch.autograd.set_detect_anomaly(True)
			model.train()
		else:
			model.eval()

		running_loss=0.0
		correct=0.0

		for j,datalist in enumerate(loader[phase],0):

			inputt=datalist
			optimizer.zero_grad()
			# t,c,v= inputt.x.size()
			# print("Num_graphs",inputt.num_graphs)
			with torch.set_grad_enabled(phase=='train'):
				
				output=model(inputt)

				# print("Outside num_graphs",output.size(0))
				# print("Output device",output.device)

				label=torch.tensor([data.y for data in datalist])
				label=(label-1).to(device)
				# print("Labels",label,"device",label.device)
				loss=criterion(output,label)

			if(phase=='train'):
				loss.backward()
				optimizer.step()
				# scheduler.step()
				# for name, param in model.named_parameters():
				# 	if(param.requires_grad):
				# 		print("Name",name,"Gradient",param.grad)

			# print("Output",output)
			predicted=torch.argmax(output,dim=1)
			# print("Predicted",predicted,"Label",label)
			total[phase]+=label.size(0)
			correct+=(predicted==label).sum().item()

			running_loss+=float(loss)*output.size(0)

			# print("Loss",loss)
			
			# writer.add_scalar('Training loss',loss,j)

		epoch_loss[phase]=running_loss/dataset_sizes[phase] 
		print("Correct",correct,"total",total[phase])
		accuracy[phase]=correct/total[phase]


		print("Epoch:",epoch,"Phase:",phase," loss:",epoch_loss[phase],"accuracy:",accuracy[phase])


		writer[phase].add_scalar('loss',epoch_loss[phase],epoch)
		writer[phase].add_scalar('accuracy',accuracy[phase],epoch)



	torch.save({'epoch':epoch,
				'model_state_dict':model.state_dict(),
				'optimmizer_state_dict':optimizer.state_dict(),
				'train_loss':epoch_loss['train'],
				'val_loss':epoch_loss['val'],
				'accuracy_train':accuracy['train'],
				'accuracy_val':accuracy['val']
				},'/ssd_scratch/cvit/nateshhariharan/checkpts/checkpt_val1/check_epoch_'+str(epoch)+'.pth')

	train_loss.append(float(epoch_loss['train']))
	val_loss.append(float(epoch_loss['val']))
	train_accuracy.append(float(accuracy['train']))
	val_accuracy.append(float(accuracy['val']))



train_loss_np=np.asarray(train_loss)
val_loss_np=np.asarray(val_loss)
train_accuracy_np=np.asarray(train_accuracy)
val_accuracy_np=np.asarray(val_accuracy)

np.save('./graph_values/expt_val1/train_loss.npy',train_loss_np)
np.save('./graph_values/expt_val1/val_loss.npy',val_loss_np)
np.save('./graph_values/expt_val1/train_accuracy.npy',train_accuracy_np)
np.save('./graph_values/expt_val1/val_accuracy.npy',val_accuracy_np)
	



