import numpy as np 
import os

path='/ssd_scratch/cvit/nateshhariharan/Dataset_NTU_RGBD/numpy_files/'

files=os.listdir(path)

def auto_padding(data,max_frame):

	M,T,V,C = data.shape

	if(T==max_frame):
		return data 

	data_new = np.zeros((M,max_frame,V,C))
	data_new[:,0:T,:,:] = data 
	return data_new 

# data_list=[]
action_list=[]
# files = files[0:20]

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

		print("Keys",keys)

		shape=data['skel_body0'].shape

		if('skel_body1' in keys):
			print("2 member data")
			data_new=np.empty((2,shape[0],shape[1],shape[2]))
			data_new[0,:,:,:]=data['skel_body0']
			data_new[1,:,:,:]=data['skel_body1']
			data_new = auto_padding(data_new,300)
			print("Shape of data",data_new.shape)
			np.save('/ssd_scratch/cvit/nateshhariharan/Dataset_NTU_RGBD/ntu60_numpy/'+name+'.npy',data_new)


		else:
			print("1 member data")
			data_new=np.zeros((2,shape[0],shape[1],shape[2]))
			data_new[0,:,:,:]=data['skel_body0']
			# data_new[1,:,:,:]=data['skel_body0']
			# print("Data original ",data_new)
			data_new = auto_padding(data_new,300)
			# print("After padding",data_new)
			# print("Shape of data",data_new.shape)
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