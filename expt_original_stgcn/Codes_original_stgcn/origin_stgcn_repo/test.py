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
from feeder.feeder import Feeder_seqlen 
import torch.nn as nn
from torch.nn import DataParallel

import numpy as np 
from sklearn.metrics import confusion_matrix

parser=argparse.ArgumentParser()

parser.add_argument("--path_test_data")
parser.add_argument("--path_test_label")
parser.add_argument("--path_test_seqlen")
parser.add_argument("--checkpt_path")
parser.add_argument("--batch_size",type=int)

args = parser.parse_args()

test_feeder = Feeder_seqlen(args.path_test_data,args.path_test_label,args.path_test_seqlen)

loader = DataLoader(dataset=test_feeder,batch_size=args.batch_size,shuffle=False,num_workers=4*2)


model = Model(3,60,graph_args=['ntu-rgb+d','spatial'])

if(torch.cuda.device_count()>1):
	print("Number of gpus",torch.cuda.device_count())
	model=DataParallel(model,device_ids=[0,1],output_device=torch.device('cuda').index)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

checkpoint=torch.load(args.checkpt_path)

model.load_state_dict(checkpoint['model_state_dict'])


model.eval()

bin_dict={'0-50':0.0,'50-100':0.0,'100-150':0.0,'150-200':0.0,'200-250':0.0,'250-300':0.0} #'300-350':0.0,'350-400':0.0,'400-450':0.0,'450-500':0.0,'500-550':0.0}
tot_bin_dict={'0-50':0.0,'50-100':0.0,'100-150':0.0,'150-200':0.0,'200-250':0.0,'250-300':0.0}#'300-350':0.0,'350-400':0.0,'400-450':0.0,'450-500':0.0,'500-550':0.0}


def histo_create(lenth_list,predcorr_list):

	for i in range(len(lenth_list)):

		if(lenth_list[i]<50):
			bin_dict['0-50']+=predcorr_list[i]
			tot_bin_dict['0-50']+=1
			# length_bin_dict['0-50'][0]+=1
			# length_bin_dict['0-50'][1]+=predcorr_list[i]

		elif(50<=lenth_list[i]<100):
			bin_dict['50-100']+=predcorr_list[i]
			tot_bin_dict['50-100']+=1
			# length_bin_dict['50-100'][0]+=1
			# length_bin_dict['50-100'][1]+=predcorr_list[i]

		elif(100<=lenth_list[i]<150):
			bin_dict['100-150']+=predcorr_list[i]
			tot_bin_dict['100-150']+=1
			# length_bin_dict['100-150'][0]+=1
			# length_bin_dict['100-150'][1]+=predcorr_list[i]

		elif(150<=lenth_list[i]<200):
			bin_dict['150-200']+=predcorr_list[i]
			tot_bin_dict['150-200']+=1
			# length_bin_dict['150-200'][0]+=1
			# length_bin_dict['150-200'][1]+=predcorr_list[i]

		elif(200<=lenth_list[i]<250):
			bin_dict['200-250']+=predcorr_list[i]
			tot_bin_dict['200-250']+=1
			# length_bin_dict['200-250'][0]+=1
			# length_bin_dict['200-250'][1]+=predcorr_list[i]

		elif(250<=lenth_list[i]<300):
			bin_dict['250-300']+=predcorr_list[i]
			tot_bin_dict['250-300']+=1
			# length_bin_dict['250-300'][0]+=1
			# length_bin_dict['250-300'][1]+=predcorr_list[i]

correct=0.0
total=0.0

labellist=torch.zeros(0,dtype=torch.long,device='cpu')
predlist=torch.zeros(0,dtype=torch.long,device='cpu')

lenth_list=torch.zeros(0,dtype=torch.long,device='cpu')
predcorr_list=torch.zeros(0,dtype=torch.long,device='cpu')


for data,label,seq_len in loader:

	inputt = data 

	output = model(inputt)

	label =label.to(device)

	predicted = torch.argmax(output,dim=1)


	lenth_list=torch.cat([lenth_list,seq_len])
	predcorr_list=torch.cat([predcorr_list,(predicted==label).cpu()])

	correct+=(predicted==label).sum().item()

	total+=len(label)

	predlist=torch.cat([predlist,predicted.view(-1).cpu()])
	labellist=torch.cat([labellist,label.view(-1).cpu()])




accuracy=correct/total
print("Total files",total)
print("Accuracy of test data",accuracy)

# print("Maximum length",max_len)
# print("Minimum length",min_len)

print("Length list",lenth_list)
print("Pred corr list",predcorr_list)

histo_create(lenth_list.numpy(),predcorr_list.numpy())

print("Bins dictionary",bin_dict)
print("Total bin dictionary",tot_bin_dict)
dict_list = [tot_bin_dict,bin_dict]
dict_np = np.asarray(dict_list)
np.save('./graph_values/length_dict_np',dict_np)


conf_mat=confusion_matrix(labellist.numpy(),predlist.numpy())

print("Confusion matrix",conf_mat)
class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
print("Number of sequences per class",conf_mat.sum(1))
np.save('./graph_values/perclass_seq.npy',np.asarray(conf_mat.sum(1)))
print("Number of correctly classified sequences per class",conf_mat.diagonal())
np.save('./graph_values/correctclass_seq.npy',np.asarray(conf_mat.diagonal()))
print("Per class accuracy",class_accuracy)
print("Size of matrix",conf_mat.shape)
print("Actual correct",conf_mat.diagonal().sum())
