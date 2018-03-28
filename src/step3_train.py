import cv2
import h5py

from step2_UNET import unet_model
from step1_preprocess import *
from keras.utils import plot_model
from matplotlib import pyplot as plt

def load_data(*num):
	imagePaths = glob.glob('{}*.pkl.gz'.format(SAVE_FOLDER_lung_mask_))
	if len(num)==1:
		imagePaths=imagePaths[:num[0]]
	elif len(num)>1:
		raise ValueError('too many variables')
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

		img = (l_slice_image*l_slice_lung_mask).transpose() #transpose
		x_min,x_max = minmax(np.where(img.sum(axis=1)!=0)[0])
		y_min,y_max = minmax(np.where(img.sum(axis=0)!=0)[0])
		imgSub = img[x_min:x_max,y_min:y_max]# get the subimage
		imgSubResize=cv2.resize(imgSub,(128,128)) #resize to 200*200
		x.append(imgSubResize[np.newaxis,:]) # convert (200,200) to (1,200,200)

		img = l_slice_nodule_mask.transpose() #transpose
		imgSub = img[x_min:x_max,y_min:y_max]# get the subimage
		imgSubResize=cv2.resize(imgSub,(128,128)) #resize to 200*200
		y.append(imgSubResize[np.newaxis,:]) # convert (200,200) to (1,200,200)
	x=np.array(x)
	y=np.array(y)
	

	np.random.seed(1) #make sure get same result
	chooselist=np.random.permutation(range(len(imagePaths))) #shuffle
	cutNum=int(np.ceil(len(imagePaths)*0.9))   # 0.1 for test 
	train_list = chooselist[:cutNum]
	test_list = chooselist[cutNum:]


	return x[train_list],y[train_list],x[test_list],y[test_list]
def minmax(array):
	return array.min(),array.max()


if __name__ == '__main__':
	x_train,y_train,x_test,y_test=load_data(20)
	print len(x_train)
	print x_train[0].shape
	# for i in range(5):
	# 	plt.figure(1)
	# 	plt.subplot(121)
	# 	plt.imshow(x_train[i][0], cmap=plt.cm.gray)
	# 	plt.subplot(122)
	# 	plt.imshow(y_train[i][0], cmap=plt.cm.gray)
	# 	plt.show()
	model = unet_model()
	history=model.fit(x_train,y_train,batch_size=4,epochs=20,verbose=2,validation_split=0.1) #0.1 for validate
    # to do : save histoty,plot history
	f=h5py.File("Unet-test_history.h5","w")
	f['dice_coef']=history.history['dice_coef']
	f['val_dice_coef']=history.history['val_dice_coef']
	f.close

	loss,accuracy = model.evaluate(x_test,y_test)
	print('\ntest loss',loss)
	print('dice_coef',accuracy)
	#save then delete model
	model.save('Unet-test.h5')
	del model
