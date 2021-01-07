import torch
import os 
import argparse 

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from net.st_gcn import Model
from torch.utils.data import DataLoader

import torchlight_new
from torchlight_new import str2bool
from torchlight_new import DictAction
from torchlight_new import import_class
from feeder.feeder import Feeder 
import torch.nn as nn
from torch.nn import DataParallel

import numpy as np 

parser=argparse.ArgumentParser()

parser.add_argument("--path_train_data")
parser.add_argument("--path_train_label")
parser.add_argument("--path_val_data")
parser.add_argument("--path_val_label")
parser.add_argument("--epochs",type=int)
parser.add_argument("--batch_size",type=int)

args = parser.parse_args()

train_feeder = Feeder(args.path_train_data,args.path_train_label)
val_feeder = Feeder(args.path_val_data,args.path_val_label)

loader = {'train':DataLoader(dataset=train_feeder,batch_size=args.batch_size,shuffle=True,drop_last=True,num_workers=4*2),
	      'val': DataLoader(dataset=val_feeder,batch_size=args.batch_size,shuffle=False,num_workers=4*2)}


model = Model(3,60,graph_args=['ntu-rgb+d','spatial'])

if(torch.cuda.device_count()>1):
	print("Number of gpus",torch.cuda.device_count())
	model=DataParallel(model,device_ids=[0,1],output_device=torch.device('cuda').index)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

writer={'train':SummaryWriter('./runs/expt_2/train'),'val':SummaryWriter('./runs/expt_2/val')}
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=0.0001)
scheduler=lr_scheduler.MultiStepLR(optimizer,milestones=[10,50],gamma=0.1)

train_loss = []
train_accuracy =[]
val_loss = []
val_accuracy = []

for epoch in range(args.epochs):

	epoch_loss={'train':0.0,'val':0.0}
	total={'train':0.0,'val':0.0}
	# correct={'train':0.0,'val':0.0}
	accuracy={'train':0.0,'val':0.0}

	for phase in loader.keys():

		print("Phase",phase)

		if(phase=='train'):
			model.train()
		else:
			model.eval()

		running_loss=0.0
		correct=0.0
		length = 0

		for data,label in loader[phase]:

			inputt=data
			length = length + len(label)
			optimizer.zero_grad()
			with torch.set_grad_enabled(phase=='train'):
				
				output=model(inputt)

				label = label.to(device)
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

		print("Length",length)

		epoch_loss[phase]=running_loss/total[phase] 
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
				},'/ssd_scratch/cvit/nateshhariharan/checkpts/checkpt_val/check_epoch_'+str(epoch)+'.pth')

	train_loss.append(float(epoch_loss['train']))
	val_loss.append(float(epoch_loss['val']))
	train_accuracy.append(float(accuracy['train']))
	val_accuracy.append(float(accuracy['val']))
	



train_loss_np=np.asarray(train_loss)
val_loss_np=np.asarray(val_loss)
train_accuracy_np=np.asarray(train_accuracy)
val_accuracy_np=np.asarray(val_accuracy)

np.save('./graph_values/expt/train_loss.npy',train_loss_np)
np.save('./graph_values/expt/val_loss.npy',val_loss_np)
np.save('./graph_values/expt/train_accuracy.npy',train_accuracy_np)
np.save('./graph_values/expt/val_accuracy.npy',val_accuracy_np)

