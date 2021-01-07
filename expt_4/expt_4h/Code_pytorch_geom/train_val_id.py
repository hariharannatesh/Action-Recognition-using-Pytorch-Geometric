import os
import torch
import shutil

path_train='/ssd_scratch/cvit/nateshhariharan/Dataset_NTU_RGBD/train_ntu60_graph/'

path_val='/ssd_scratch/cvit/nateshhariharan/Dataset_NTU_RGBD/val_ntu60_graph/'

files=os.listdir(path_train)

val_ids=["035","027","018","017","014"]

for f in files:

	if(f.split('P')[1].split('R')[0] in val_ids):
		file_val=os.path.join(path_val,f)
		file_train=os.path.join(path_train,f)
		shutil.move(file_train,file_val)

