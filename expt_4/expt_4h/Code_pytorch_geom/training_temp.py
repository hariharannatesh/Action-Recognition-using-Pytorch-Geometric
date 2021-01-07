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
from geometric_stgcn import Model 

from stride_geometric import Sample_batch
from graph_2 import Graph

import argparse

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import DataParallel
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
# torch.set_printoptions(precision=10)
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

# print("Edge index sp",edge_index_sp)



# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loader={'train':DataListLoader([torch.load(os.path.join(args.path_train,f)) for f in files_t],batch_size=args.batch_size,shuffle=True,num_workers=4*2),
		'val':DataListLoader([torch.load(os.path.join(args.path_val,f)) for f in files_v],batch_size=args.batch_size,shuffle=True,num_workers=4*2)}

print("Loader train:",len(loader['train']),"Val train",len(loader['val']))

dataset_sizes={'train':len(files_t),'val':len(files_v)}
print("Dataset sizes",dataset_sizes)

model=Model(3,256,num_classes=60).double()

if(torch.cuda.device_count()>1):
	print("Number of gpus",torch.cuda.device_count())
	model=DataParallel(model,output_device=torch.device('cuda').index)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# print("Device",device)

writer={'train':SummaryWriter('./runs/expt_val1/train'),'val':SummaryWriter('./runs/expt_val1/val')}
# epochs=1
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.05,momentum=0.9,weight_decay=0.0001)
# scheduler=lr_scheduler.StepLR(optimizer,step_size=80,gamma=0.96)
scheduler=lr_scheduler.MultiStepLR(optimizer,milestones=[20,60],gamma=0.4)
# print("Model parameters",list(model.named_parameters()))
train_loss=[]
train_accuracy=[]
val_loss=[]
val_accuracy=[]
lr_list = []

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
				for param_group in optimizer.param_groups:
					print (param_group['lr'])
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
		if(phase=='train'):
			scheduler.step()



	torch.save({'epoch':epoch,
				'model_state_dict':model.state_dict(),
				'optimmizer_state_dict':optimizer.state_dict(),
				'train_loss':epoch_loss['train'],
				'val_loss':epoch_loss['val'],
				'accuracy_train':accuracy['train'],
				'accuracy_val':accuracy['val']
				},'/ssd_scratch/cvit/nateshhariharan/checkpts/checkpt_val1h/check_epoch_'+str(epoch)+'.pth')

	train_loss.append(float(epoch_loss['train']))
	val_loss.append(float(epoch_loss['val']))
	train_accuracy.append(float(accuracy['train']))
	val_accuracy.append(float(accuracy['val']))
	for param_group in optimizer.param_groups:
		lr_list.append(float(param_group['lr']))
	



train_loss_np=np.asarray(train_loss)
val_loss_np=np.asarray(val_loss)
train_accuracy_np=np.asarray(train_accuracy)
val_accuracy_np=np.asarray(val_accuracy)
lr_np = np.asarray(lr_list)

np.save('./graph_values/expt_val1/train_loss.npy',train_loss_np)
np.save('./graph_values/expt_val1/val_loss.npy',val_loss_np)
np.save('./graph_values/expt_val1/train_accuracy.npy',train_accuracy_np)
np.save('./graph_values/expt_val1/val_accuracy.npy',val_accuracy_np)
np.save('./graph_values/expt_val1/lr.npy',lr_np)