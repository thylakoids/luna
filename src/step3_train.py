from step2_UNET import unet_model
from step1_preprocess import *
from keras.utils import plot_model
from matplotlib import pyplot as plt

def load_data():
	imagePaths = glob.glob('{}*.pkl.gz'.format(SAVE_FOLDER_lung_mask_))
	x=[]
	y=[]
	for imagePath in imagePaths:
		file = gzip.open(imagePath,'rb')
		l_slice_lung_mask = pickle.load(file)
		file.close()

		file = gzip.open(imagePath.replace(SAVE_FOLDER_lung_mask,SAVE_FOLDER_image))
		l_slice_image= pickle.load(file)
		file.close()

		file = gzip.open(imagePath.replace(SAVE_FOLDER_lung_mask,SAVE_FOLDER_nodule_mask))
		l_slice_nodule_mask= pickle.load(file)
		file.close()
		x.append(l_slice_image*l_slice_lung_mask[np.newaxis,:]) #??????
		y.append(l_slice_nodule_mask[np.newaxis,:])
	x=np.array(x)
	y=np.array(y)
	return x[:10],y[:10],x[10:20],y[10:20]

if __name__ == '__main__':
	x_train,y_train,x_test,y_test=load_data()
	print len(x_train)
	print x_train[0].shape
	for i in range(5):
		plt.figure(1)
		plt.subplot(121)
		plt.imshow(x_train[i][0], cmap=plt.cm.gray)
		plt.subplot(122)
		plt.imshow(y_train[i][0], cmap=plt.cm.gray)
		plt.show()
	model = unet_model()
	# model.fit(x_train,y_train,batch_size=1,epochs=20,verbose=2,validation_data=(x_test,y_test))


