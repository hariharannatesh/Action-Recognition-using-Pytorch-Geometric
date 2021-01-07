import torch 
import torch.nn as nn
from torch_geometric.data import DataListLoader
from torch_geometric.nn import DataParallel
from torch_geometric.data import Batch
from geometric_stgcn import Model
import numpy as np 

import argparse
import os
from sklearn.metrics import confusion_matrix

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()

parser.add_argument("--path_test")
parser.add_argument("--checkpt_path")
parser.add_argument("--batch_size",type=int)

args=parser.parse_args()
files_test=os.listdir(args.path_test)
print("Number of files in test",len(files_test))

test_loader=DataListLoader([torch.load(os.path.join(args.path_test,f)) for f in files_test],batch_size=args.batch_size)


model=Model(3,256,num_classes=60).double()

if(torch.cuda.device_count()>1):
	print("Number of gpus",torch.cuda.device_count())
	model=DataParallel(model,device_ids=[0,1],output_device=torch.device('cuda').index)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

checkpoint=torch.load(args.checkpt_path)

model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

correct=0.0
total=0.0

labellist=torch.zeros(0,dtype=torch.long,device='cpu')
predlist=torch.zeros(0,dtype=torch.long,device='cpu')

lenth_list=torch.zeros(0,dtype=torch.long,device='cpu')
predcorr_list=torch.zeros(0,dtype=torch.long,device='cpu')

max_len=0.0
min_len=100000

# bins=[0,50,100,150,200,250,300,350,400,450,500]
bin_dict={'0-50':0.0,'50-100':0.0,'100-150':0.0,'150-200':0.0,'200-250':0.0,'250-300':0.0} #'300-350':0.0,'350-400':0.0,'400-450':0.0,'450-500':0.0,'500-550':0.0}
tot_bin_dict={'0-50':0.0,'50-100':0.0,'100-150':0.0,'150-200':0.0,'200-250':0.0,'250-300':0.0}#'300-350':0.0,'350-400':0.0,'400-450':0.0,'450-500':0.0,'500-550':0.0}
# length_bin_dict={'0-50':[0.0,0.0],'50-100':[0.0,0.0],'100-150':[0.0,0.0],'150-200':[0.0,0.0],'200-250':[0.0,0.0],'250-300':[0.0,0.0],'300-350':[0.0,0.0],'350-400':[0.0,0.0],'400-450':[0.0,0.0],'450-500':[0.0,0.0],'500-550':[0.0,0.0]}
accuracy_bin_dict={'0-50':0.0,'50-100':0.0,'100-150':0.0,'150-200':0.0,'200-250':0.0,'250-300':0.0,'300-350':0.0,'350-400':0.0,'400-450':0.0,'450-500':0.0,'500-550':0.0}

total_length_bin_dict={}
correct_bin_dict={}

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

		# elif(300<=lenth_list[i]<350):
		# 	bin_dict['300-350']+=predcorr_list[i]
		# 	tot_bin_dict['300-350']+=1
		# 	length_bin_dict['300-350'][0]+=1
		# 	length_bin_dict['300-350'][1]+=predcorr_list[i]

		# elif(350<=lenth_list[i]<400):
		# 	bin_dict['350-400']+=predcorr_list[i]
		# 	tot_bin_dict['350-400']+=1
		# 	length_bin_dict['350-400'][0]+=1
		# 	length_bin_dict['350-400'][1]+=predcorr_list[i]


		# elif(400<=lenth_list[i]<450):
		# 	bin_dict['400-450']+=predcorr_list[i]
		# 	tot_bin_dict['400-450']+=1
		# 	length_bin_dict['400-450'][0]+=1
		# 	length_bin_dict['400-450'][1]+=predcorr_list[i]

		# elif(450<=lenth_list[i]<500):
		# 	bin_dict['450-500']+=predcorr_list[i]
		# 	tot_bin_dict['450-500']+=1
		# 	length_bin_dict['450-500'][0]+=1
		# 	length_bin_dict['450-500'][1]+=predcorr_list[i]


		# elif(500<=lenth_list[i]<550):
		# 	bin_dict['500-550']+=predcorr_list[i]
		# 	tot_bin_dict['500-550']+=1
		# 	length_bin_dict['500-550'][0]+=1
		# 	length_bin_dict['500-550'][1]+=predcorr_list[i]


	# for key in bin_dict.keys():
	# 	accuracy_bin_dict[key]=100.0*(bin_dict[key]/tot_bin_dict[key])




for j,datalistt in enumerate(test_loader,0):

	inputt=datalistt

	batched_data=Batch.from_data_list(datalistt)
	batch_index=batched_data.batch 
	# __,length_seq=torch.unique(batch_index,return_counts=True)
	length_seq = batched_data['seq_len']

	maxi=torch.max(length_seq)
	mini=torch.min(length_seq)

	if(maxi>max_len):
		max_len=maxi 

	if(mini<min_len):
		min_len=mini

	output=model(inputt)

	label=torch.tensor([data.y for data in datalistt])
	label=(label-1).to(device)


	predicted=torch.argmax(output,dim=1)
	total+=label.size(0)

	lenth_list=torch.cat([lenth_list,length_seq])
	predcorr_list=torch.cat([predcorr_list,(predicted==label).cpu()])

	correct+=(predicted==label).sum().item()

	predlist=torch.cat([predlist,predicted.view(-1).cpu()])
	labellist=torch.cat([labellist,label.view(-1).cpu()])


accuracy=correct/total
print("Total files",total)
print("Accuracy of test data",accuracy)

print("Maximum length",max_len)
M = max_len
print("Minimum length",min_len)
m = min_len

keys = range(m,M+1)

for i in keys:
	total_length_bin_dict[i]=0.0
	correct_bin_dict[i]=0.0


print("Length list",lenth_list)
print("Pred corr list",predcorr_list)
print("Before tot length dict",total_length_bin_dict)
print("Before correct length dict",correct_bin_dict)
# histo_create(lenth_list.numpy(),predcorr_list.numpy())


def histo_create_bin1(lenth_list,predcorr_list):

	for i,j in zip(lenth_list,predcorr_list):

		total_length_bin_dict[i]+=1
		correct_bin_dict[i]+=j


# print("Bins dictionary",bin_dict)
# print("Total bin dictionary",tot_bin_dict)
# dict_list = [tot_bin_dict,bin_dict]
# dict_np = np.asarray(dict_list)
# length_dict = np.asarray(length_bin_dict)
# np.save('./graph_values/length_dict_np_2',length_dict)
# np.save('./graph_values/length_dict_np',dict_np)
# print("Accuracy bin dictionary",accuracy_bin_dict)

histo_create_bin1(lenth_list.numpy(),predcorr_list.numpy())

print("Total length bins",total_length_bin_dict)
print("Correct classified bins",correct_bin_dict)
dict_list_new = [total_length_bin_dict,correct_bin_dict]
dict_np_new = np.asarray(dict_list_new)
np.save('./graph_values/length_dict_bin1_np.npy',dict_np_new)

# fig1=plt.figure(figsize=(10,8))
# plt.bar(list(bin_dict.keys()),list(bin_dict.values()),color='maroon',width=0.4)
# # plt.xticks([0,50,100,150,200,250,300,350,400,450])
# plt.xlabel("Range of length of sequences ")
# plt.ylabel("Number of correctly classified sequences per range")
# plt.savefig("histogram.png",dpi=400)
# plt.close()


# fig2=plt.figure(figsize=(10,8))
# plt.bar(list(tot_bin_dict.keys()),list(tot_bin_dict.values()),color='blue',width=0.4)
# # plt.xticks([0,50,100,150,200,250,300,350,400,450])
# plt.xlabel("Range of length of sequences ")
# plt.ylabel("Total number of sequences per range")
# plt.savefig("total_histogram.png",dpi=400)
# plt.close()

# fig3=plt.figure(figsize=(10,8))
# plt.bar(list(accuracy_bin_dict.keys()),list(accuracy_bin_dict.values()),color='green',width=0.4)
# plt.xlabel("Range of length of sequences ")
# plt.ylabel("Percentage of correctly classified sequences per range")
# plt.savefig("accuracy_histogram.png",dpi=400)

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






