import numpy as np 
from keras import backend as K 

def get_model_memory_usage(batch_size,model):
	activations = 0
	for l in model.layers:
		single_layer = 1
		for s in l.output_shape:
			if s is None:
				continue
			single_layer *= s
		activations += single_layer

	parameters = np.sum([K.count_params(p) for p in model.weights])


	activations_memory = 4.0*batch_size*activations*2 # and gradients
	parameters_memory = 4.0*parameters*3 # and gradients and cache
	total_memory = activations_memory + parameters_memory
	gbytes = [ np.round(mem / (1024.0 ** 3), 3)  for mem in [activations_memory,parameters_memory,total_memory]]

	print 'estimated total memory usage: {}GB\nactivations: {}\nactivations memory: {}GB\nparameters: {}\nparameters memory: {}GB\n'.format(gbytes[2],activations,gbytes[0],parameters,gbytes[1])
	
if __name__ == '__main__':
	import sys
	sys.path.append("..")
	from step2_UNET import unet_model
	model = unet_model()
	get_model_memory_usage(2,model)
