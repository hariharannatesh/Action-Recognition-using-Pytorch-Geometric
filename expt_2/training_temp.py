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
from torch_sparse import SparseTensor
from norm_original import Norm

from stride_geometric import Sample_batch
from graph_2 import Graph

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
				edge_importance_weighting):
		super().__init__()
		self.inc=input_channels
		self.outc=output_channels
		self.nnode=Net_nodes(input_channels,output_channels)
		self.conv1=GCNConv(output_channels,output_channels,node_dim=1,bias=True,normalize=False)
		# self.conv1.weight=nn.Parameter(torch.ones_like(self.conv1.weight,dtype=torch.float64))
		nn.init.xavier_normal_(self.conv1.weight,gain=nn.init.calculate_gain('relu'))
		nn.init.zeros_(self.conv1.bias)
		self.conv1.weight=nn.Parameter(self.conv1.weight.type(torch.float64))
		self.gn=Norm('gn',output_channels*25)
		self.s=Sample_batch()

		if(edge_importance_weighting==True):
			self.edge_importance=nn.Parameter(torch.ones(adj_old.size()))
		else:
			self.edge_importance=torch.tensor(1)

		# print("Time conv weights",self.conv1.weight)

	def forward(self,data,stride=1,kernel=9,residual=False):

		x,edge_index_t=data.x,data.edge_index
		data_final=data

		adj_old_new=adj_old.to(x.device)

		A=adj_old_new * self.edge_importance

		edge_index_sp=[]
		for i in range(len(A)):
			edge_index_sp.append(SparseTensor.from_dense(A[i]))


		# print("Num_graphs:",data.num_graphs)

		x_res=torch.tensor(0)
		# x_res=x_res.to(device)

		if(residual):
			if(self.inc==self.outc):
				x_res=x
				# x_res=x_res.type(torch.float32) 
				# x_res=torch.tanh(x_res) #not in original stgcn
				# data_temp=data
				# data_temp=self.gn(data_temp,data_temp['x'])
				# x_res=data_temp['x']
				# x_res=torch.tanh(x_res)
				# x_res=F.sigmoid(x_res)

			else:
				# weights=nn.Parameter(torch.randn(x.size(0),self.inc,self.outc,dtype=torch.float32,device=device))
				weights=nn.Parameter(torch.empty(x.size(0),self.inc,self.outc,dtype=torch.float32)).cuda()
				weight=nn.init.xavier_normal_(weights,gain=nn.init.calculate_gain('relu')).cuda()
				x_res=x
				x_res=x_res.type(torch.float32)
				x_res=torch.matmul(x_res,weights)
				data_res=data
				data_res.x=x_res
				if(stride!=1):
					data_res=self.s.forward(data_res,kernel,stride) 
				# print("Residual data device",data_res['x'].device)
				data_res=self.gn(data_res,data_res['x'])
				x_res=data_res.x
				# x_res=torch.tanh(x_res) #not in original stgcn
				# x_res=F.sigmoid(x_res)


		else:
			x_res=torch.tensor(0)
			# x_res=torch.tanh(x_res)
			# x_res=x_res.to(device)

		edge_weight=torch.ones((edge_index_t.size(1), ),dtype=torch.long).cuda()


		x=self.nnode(x=x,edge_index=edge_index_sp)
		# x=self.nnode(x=x,edge_index=edge_index_og)
		# print("After spatial conv",x)
		# print(edge_index_t)

		edge_index_t,edge_weight=utils.add_self_loops(edge_index_t,edge_weight)
		# print("X device",x.device)
		data=self.gn(data,x)
		x=F.relu(data.x)
		# x=torch.tanh(data.x) #works for now

		edge_index_t=edge_index_t.cuda()
		# print("Edge index time device",edge_index_t.device)

		x=x.permute(1,0,2).contiguous()
		x=self.conv1(x,edge_index_t,edge_weight=edge_weight)
		x=x.permute(1,0,2).contiguous()

		data.x = x
		data=self.gn(data,x)
		data.x=F.dropout(data.x,p=0.5,training=self.training,inplace=True)
		# x=F.relu(data.x)
		# x=torch.tanh(data.x)

		data_final['x']=x

		if(stride!=1):
			data_final=self.s.forward(data_final,kernel,stride)

		# print("Data final device",data_final.x.device," x_res device",x_res.device)
		data_final.x = data_final.x + x_res
		data_final.x=F.relu(data_final.x)
		# data_final.x=F.sigmoid(data_final.x)
		# data_final.x=torch.tanh(data_final.x)

		return data_final



class Net_nodes(nn.Module):
	
	def __init__(self,
				input_channels,
				output_channels):
		super().__init__()
		self.conv1=GCNConv(input_channels,output_channels,node_dim=1,bias=True,cached=True)
		# self.conv1.weight.detach().cpu()
		self.conv1.weight=nn.Parameter(torch.empty(self.conv1.weight.size(),dtype=torch.float64))
		nn.init.xavier_normal_(self.conv1.weight,gain=nn.init.calculate_gain('relu'))
		nn.init.zeros_(self.conv1.bias)
		# print("Self conv1 weight device inside class",self.conv1.weight)

		self.conv2=GCNConv(input_channels,output_channels,node_dim=1,bias=True,cached=True)
		self.conv2.weight=nn.Parameter(torch.empty(self.conv2.weight.size(),dtype=torch.float64))
		nn.init.xavier_normal_(self.conv2.weight,gain=nn.init.calculate_gain('relu'))
		nn.init.zeros_(self.conv2.bias)
		
		self.conv3=GCNConv(input_channels,output_channels,node_dim=1,bias=True,cached=True)
		self.conv3.weight=nn.Parameter(torch.empty(self.conv3.weight.size(),dtype=torch.float64))
		nn.init.xavier_normal_(self.conv3.weight,gain=nn.init.calculate_gain('relu'))
		nn.init.zeros_(self.conv3.bias)
		

		# # self.conv1._cached_adj_t=None
		# # print("Spatial conv weights",self.conv1.weight)
		# # self.conv2=GCNConv(4,output_channels,node_dim=1)

		# self.conv={}
		# for i in range(len(edge_index_sp)):
		# 	self.conv[str(i)]=GCNConv(input_channels,output_channels,cached=True,bias=True)

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

		
		return x_sum


class Model(nn.Module):

	def __init__(self,input_channels,output_channels,num_classes):

		super().__init__()

		self.gnn=Norm('gn',150)
		self.stgcn1=Net_time(input_channels,64,edge_importance_weighting=True)
		self.stgcn2=Net_time(64,64,edge_importance_weighting=True)
		self.stgcn3=Net_time(64,64,edge_importance_weighting=True)
		self.stgcn4=Net_time(64,64,edge_importance_weighting=True)
		self.stgcn5=Net_time(64,128,edge_importance_weighting=True)
		self.stgcn6=Net_time(128,128,edge_importance_weighting=True) #commented for now
		self.stgcn7=Net_time(128,128,edge_importance_weighting=True)
		self.stgcn8=Net_time(128,256,edge_importance_weighting=True)
		# # self.stgcn9=Net_time(256,output_channels) #newly added
		self.stgcn9=Net_time(256,256,edge_importance_weighting=True)
		self.stgcn10=Net_time(256,output_channels,edge_importance_weighting=True)
		self.prediction_mat=nn.Parameter(torch.empty(output_channels,num_classes,dtype=torch.float64))
		self.prediction_mat=nn.init.xavier_normal_(self.prediction_mat,gain=1)
		self.softmax=nn.Softmax(dim=1)


	def forward(self,data):

		# print("Data before conv inside module ",data.num_graphs,data.batch.device,data.x.size())
		# print("Data y",data.y)

		data=self.gnn(data,data.x)

		op=self.stgcn1(data,kernel=9)
		# print("After first stgcn",torch.isnan(op.x).any())
		# print("Nnode param",self.stgcn1.nnode.conv[0].weight.device)
		op=self.stgcn2(op,kernel=9,residual=True)
		# print("After second stgcn",torch.isnan(op.x).any())
		op.x=op.x.type(torch.float64)
		op=self.stgcn3(op,kernel=9,residual=True)
		# print("After third stgcn",torch.isnan(op.x).any())
		op.x=op.x.type(torch.float64)
		op=self.stgcn4(op,kernel=9,residual=True)
		# print("After fourth stgcn",torch.isnan(op.x).any())
		op.x=op.x.type(torch.float64)
		op=self.stgcn5(op,kernel=9,stride=2,residual=True)
		# print("After fifth stgcn",torch.isnan(op.x).any())
		op.x=op.x.type(torch.float64)
		op=self.stgcn6(op,kernel=9,residual=True)
		# # print("After sixth stgcn",torch.isnan(op.x).any())
		op.x=op.x.type(torch.float64)
		op=self.stgcn7(op,kernel=9,residual=True)
		# # print("After seventh stgcn",torch.isnan(op.x).any())
		op.x=op.x.type(torch.float64)
		op=self.stgcn8(op,kernel=9,stride=2,residual=True)
		# # print("After eighth stgcn",torch.isnan(op.x).any())
		op.x=op.x.type(torch.float64)
		op=self.stgcn9(op,kernel=9,residual=True)
		# # print("After ninth stgcn",torch.isnan(op.x).any())
		op.x=op.x.type(torch.float64)
		op=self.stgcn10(op,kernel=9,residual=True)
		# # # print("After tenth stgcn",torch.isnan(op.x).any())
		op.x=op.x.type(torch.float64)

		# print(op.x.size())
		x_result=op.x.type(torch.float64)

		op.batch=op.batch.cuda()
		# print("X_result device",x_result.device)

		# print("Batch device",op.batch.device)

		x_avg=GMP(x_result,batch=op.batch)

		x_mean=x_avg.mean(dim=1)

		# print("Avg",x_mean)

		predicted_val=torch.matmul(x_mean,self.prediction_mat)
		# print("Predicted val",predicted_val)
		# predicted_val=F.relu(predicted_val)
		# predicted_val=self.softmax(predicted_val)



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
# files_t=files_t[0:4]
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

loader={'train':DataListLoader([torch.load(os.path.join(args.path_train,f)) for f in files_t],batch_size=args.batch_size,shuffle=True),
		'val':DataListLoader([torch.load(os.path.join(args.path_val,f)) for f in files_v],batch_size=args.batch_size,shuffle=True)}

print("Loader train:",len(loader['train']),"Val train",len(loader['val']))

dataset_sizes={'train':len(files_t),'val':len(files_v)}
print("Dataset sizes",dataset_sizes)

model=Model(6,256,num_classes=60).double()
# model=Model(3,128,num_classes=60).double() #change output channels to 128
# model=Model(3,64,num_classes=60).double()
# print("Weights of spatial conv",getattr(model,"Spconv"))

if(torch.cuda.device_count()>1):
	print("Number of gpus",torch.cuda.device_count())
	model=DataParallel(model,output_device=torch.device('cuda').index)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# print("Device",device)

writer={'train':SummaryWriter('./runs/expt_val1/train'),'val':SummaryWriter('./runs/expt_val1/val')}
# epochs=1
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=0.0001)
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