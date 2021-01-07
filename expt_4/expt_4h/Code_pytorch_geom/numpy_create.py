import numpy as np 
import os

path='/ssd_scratch/cvit/nateshhariharan/Dataset_NTU_RGBD/numpy_files/'

files=os.listdir(path)

# data_list=[]
action_list=[]

for f in files:

	# print("File name",f)

	classs=f.split('C')[0].split('S')[1]
	classs=int(classs)

	name=f.split('.')[0]
	print('Name',name)
	if (classs<=17):
		data=np.load(os.path.join(path,f),allow_pickle=True).item()

		action=f.split('A')[1].split('.')[0]

		action=int(action)

		action_list.append(action)

		keys=data.keys()

		shape=data['skel_body0'].shape

		if('skel_body1' in keys):
			data_new=np.empty((2,shape[0],shape[1],shape[2]))
			data_new[0,:,:,:]=data['skel_body0']
			data_new[1,:,:,:]=data['skel_body1']
			print("Shape of data",data_new.shape)
			np.save('/ssd_scratch/cvit/nateshhariharan/Dataset_NTU_RGBD/ntu60_numpy/'+name+'.npy',data_new)


		else:
			data_new=np.empty((1,shape[0],shape[1],shape[2]))
			data_new[0,:,:,:]=data['skel_body0']
			print("Shape of data",data_new.shape)
			np.save('/ssd_scratch/cvit/nateshhariharan/Dataset_NTU_RGBD/ntu60_numpy/'+name+'.npy',data_new)

	else:
		continue


# data_np=np.asarray(data_list)
# action_list=np.asarray(action_list)
# np.save('/ssd_scratch/cvit/nateshhariharan/nturgbd_skeletons_s001_to_s017/action_list.npy',action_list)
# np.save('/ssd_scratch/cvit/nateshhariharan/nturgbd_skeletons_s001_to_s017/temp_list/data_temporal_added.npy',data_np)

# d=np.load('/ssd_scratch/cvit/nateshhariharan/nturgbd_skeletons_s001_to_s017/temp_list/data_temporal_added.npy')

# print(d[2].shape)









	# print("Keys",keys)