import glob
import numpy as np
import h5py
import random

import scipy.ndimage as nd

from utils.pathname import *
from utils.xyz import load_slice
from utils.normalize import normalizePlanes, zero_center
from utils.visualization import impose
from UNET import unet_model,dice_coef,dice_coef_loss
from matplotlib import pyplot as plt

# from utils.myMultiGpu import multi_gpu_model
from keras.utils import multi_gpu_model
from keras.optimizers import Adam,SGD
from keras.models import load_model

from config import conf


def getcroprange(lung_mask):
    pos = np.where(lung_mask >0)
    x = pos[0].min()
    y = pos[1].min()
    h = pos[0].max()-pos[0].min()
    w = pos[1].max()-pos[1].min()  
    return x,y,h,w
def crop(lung,nodule_mask,xpos,ypos,h,w):

    lung = lung[xpos:xpos+h,ypos:ypos+w]
    nodule_mask = nodule_mask[xpos:xpos+h,ypos:ypos+w]
    lung = nd.interpolation.zoom(lung, [512.0/h,512.0/w],order=0)
    nodule_mask = nd.interpolation.zoom(nodule_mask, [512.0/h,512.0/w],order=0)
    return lung,nodule_mask

def generate_data_from_file(paths):
    random.seed(10)
    PositiveImagePaths=[]
    NegativeImagePaths=[]
    for path in paths:
        PositiveImagePaths.extend(glob.glob(os.path.join(path,'*.+z*pkl.gz')))
        NegativeImagePaths.extend(glob.glob(os.path.join(path,'*.-z*pkl.gz')))
    random.shuffle(PositiveImagePaths)
    random.shuffle(NegativeImagePaths)

    batchSize=conf.BATCHSIZE
    i=0
    nSample=len(PositiveImagePaths)
    while 1:
        imagePaths =[PositiveImagePaths,NegativeImagePaths]
        x=[]
        y=[]
        for j in range(batchSize/2):
            for k in range(1):#only positive
                lung, lung_mask, nodule_mask, _,_ =load_slice(imagePaths[k][(i+j)%nSample])
                lung = normalizePlanes(lung)
                # lung = zero_center(lung)
                lung_mean = lung[lung_mask==1].mean()
                lung_std = lung[lung_mask==1].std()

                lung[lung_mask==0] = lung_mean-1.2*lung_std
                lung = lung - lung_mean
                lung = lung/lung_std

                xpos,ypos,h,w=getcroprange(lung_mask)
                lung,nodule_mask = crop(lung,nodule_mask,xpos,ypos,h,w)
                x.append(lung[np.newaxis,:])
                y.append(nodule_mask[np.newaxis,:])
        i+=batchSize/2
        i=i%nSample
        yield np.array(x),np.array(y)
def main(gpus=2):
    contains = ['subset{}'.format(i) for i in range(8)]
    contains_val = ['subset{}'.format(8)]
    contains_test = ['subset{}'.format(9)]
    train_data=generate_data_from_file([slices_folder(contain) for contain in contains])
    validation_data = generate_data_from_file([slices_folder(contain) for contain in contains_val])
    test_data = generate_data_from_file([slices_folder(contain) for contain in contains_test])

    #init model
    model = unet_model()
    if gpus>=2:
        gpu_model = multi_gpu_model(model,gpus=gpus)
    model.summary()
    gpu_model.summary()
    sgd = SGD(lr=0.1,decay = 1e-6,momentum=0.9,nesterov=True)
    adam = Adam(lr=1.0e-5)
    gpu_model.compile(optimizer=sgd, loss=dice_coef_loss, metrics=[dice_coef])
    #trian model
    steps=conf.STEPS
    history=gpu_model.fit_generator(train_data,steps_per_epoch=steps*8,epochs=conf.EPOCHS,verbose=2,
                                validation_data=validation_data,validation_steps=steps,initial_epoch=0)
    #save histoty,plot history
    f=h5py.File("Unet-history2.h5","w")
    f['dice_coef']=history.history['dice_coef']
    f['val_dice_coef']=history.history['val_dice_coef']
    f.close

    loss,accuracy = gpu_model.evaluate_generator(test_data,steps=steps)
    print('\ntest loss',loss)
    print('dice_coef',accuracy)
    #save then delete model
    model.save('Unet-model2.h5')
    del model
    del gpu_model
def tes_generator():
    contains = conf.FOLDERS
    train_data = generate_data_from_file([slices_folder(contain) for contain in contains])
    for j in range(10):
        x, y = train_data.next()
        print x.shape,y.shape
        _, plots = plt.subplots(1, 2,figsize=(10,7))
        plots[0].imshow(x[0][0],cmap='gray')
        plots[1].imshow(y[0][0],cmap='gray')
        # xy=impose(x[0][0]+1,y[0][0])/2
        # plots[2].imshow(xy)
        plt.savefig('data{}.png'.format(j))
        plt.show()
if __name__ == '__main__':
    tes_generator()
    # main()



