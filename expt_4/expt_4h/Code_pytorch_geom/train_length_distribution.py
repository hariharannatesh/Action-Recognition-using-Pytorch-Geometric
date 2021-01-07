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
# parser.add_argument("--checkpt_path")
parser.add_argument("--batch_size",type=int)

args=parser.parse_args()
files_test=os.listdir(args.path_test)
print("Number of files in test",len(files_test))

test_loader=DataListLoader([torch.load(os.path.join(args.path_test,f)) for f in files_test],batch_size=args.batch_size)


lenth_list=torch.zeros(0,dtype=torch.long,device='cpu')
max_len=0.0
min_len=100000

total_length_bin_dict={}




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


	lenth_list=torch.cat([lenth_list,length_seq])



print("Maximum length",max_len)
M = max_len
print("Minimum length",min_len)
m = min_len

keys = range(m,M+1)
for i in keys:
	total_length_bin_dict[i]=0.0

def histo_create_bin1(lenth_list):

	for i in lenth_list:

		total_length_bin_dict[i]+=1


histo_create_bin1(lenth_list.numpy())

print("Train data length distribution",total_length_bin_dict)
train_len_list = [total_length_bin_dict]
train_len_np = np.asarray(train_len_list)

np.save('./graph_values/train_length_data_distribution.npy',train_len_np)


	
