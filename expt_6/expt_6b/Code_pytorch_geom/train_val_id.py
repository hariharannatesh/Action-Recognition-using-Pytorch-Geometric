import os
import torch
import shutil

path_data='/ssd_scratch/cvit/nateshhariharan/Dataset_NTU_RGBD/ntu60_numpy/'

path_train = '/ssd_scratch/cvit/nateshhariharan/Dataset_NTU_RGBD/train_ntu60_graph/'
path_val='/ssd_scratch/cvit/nateshhariharan/Dataset_NTU_RGBD/val_ntu60_graph/'
path_test = '/ssd_scratch/cvit/nateshhariharan/Dataset_NTU_RGBD/test_ntu60_graph/'

files=os.listdir(path_data)

train_ids=["001","002","004","005","008","009","013","015","016","019","025","028","031","034","038"]

val_ids=["035","027","018","017","014"]

for f in files:

	file=os.path.join(path_data,f)

	if(f.split('P')[1].split('R')[0] in train_ids):
		file_train=os.path.join(path_train,f)
		shutil.copyfile(file,file_train)

	elif(f.split('P')[1].split('R')[0] in val_ids):
		file_val=os.path.join(path_val,f)
		shutil.copyfile(file,file_val)

	else:
		file_test = os.path.join(path_test,f)
		shutil.copyfile(file,file_test)



