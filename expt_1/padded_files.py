import numpy as np
import os

path='/ssd_scratch/cvit/nateshhariharan/Dataset_NTU_RGBD/singleperson_105_numpyfiles/'

files=os.listdir(path)

def auto_pading(data_numpy, size, random_pad=False):
    T, C, V = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((size, C, V))
        data_numpy_paded[begin:begin + T, :,:] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


for f in files:

	data=np.load(os.path.join(path,f),allow_pickle=True)
	# print(data.shape)
	data_padded= auto_pading(data,104)
	print("Data padded shape",data_padded.shape)
	np.save('/ssd_scratch/cvit/nateshhariharan/Dataset_NTU_RGBD/singleperson_105padded_npfiles/'+f,data_padded)