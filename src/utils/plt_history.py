import h5py

from matplotlib import pyplot as plt

def plt_history(filename):
	f=h5py.File(filename,"r")
	# summarize history for accuracy
	plt.plot(f['dice_coef'])
	plt.plot(f['val_dice_coef'])
	plt.title('model accuracy')
	plt.ylabel('dice_coef')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

if __name__=='__main__':
	plt_history('../model_test-history.h5')