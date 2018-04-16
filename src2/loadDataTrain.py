import glob
import numpy as np
import h5py
import random

import scipy.ndimage
from utils.pathname import slices_folder
from utils.xyz import load_slice
from utils.normalize import normalizePlanes, zero_center
from UNET import unet_model


def generate_data_from_file(paths):
    random.seed(10)
    PositiveImagePaths=[]
    NegativeImagePaths=[]
    for path in paths:
        PositiveImagePaths.extend(glob.glob('{}*.+z*pkl.gz'.format(path)))
        NegativeImagePaths.extend(glob.glob('{}*.-z*pkl.gz'.format(path)))
    random.shuffle(PositiveImagePaths)
    random.shuffle(NegativeImagePaths)

    batchSize=4
    i=0
    nSample=len(PositiveImagePaths)
    while 1:
        imagePaths =[PositiveImagePaths,NegativeImagePaths]
        x=[]
        y=[]
        for j in range(batchSize/2):
            for k in range(2):
                lung, lung_mask, nodule_mask, _,_ =load_slice(imagePaths[k][(i+j)%nSample])
                lung[lung_mask==0]=-1000
                lung = normalizePlanes(lung)
                lung = zero_center(lung)
                resize=256.0/lung.shape[1]
                lung = scipy.ndimage.zoom(lung,resize)
                nodule_mask = scipy.ndimage.zoom(lung_mask, resize)

                x.append(lung[np.newaxis,:])
                y.append(nodule_mask[np.newaxis,:])
        i+=batchSize/2
        i=i%nSample
        yield np.array(x),np.array(y)
def main():
    contains = ['subset{}'.format(i) for i in range(8)]
    contains_val = ['subset{}'.format(8)]
    contains_test = ['subset{}'.format(9)]
    train_data=generate_data_from_file([slices_folder(contain) for contain in contains])
    validation_data = generate_data_from_file([slices_folder(contain) for contain in contains_val])
    test_data = generate_data_from_file([slices_folder(contain) for contain in contains_test])

    model = unet_model()
    steps=1000
    history=model.fit_generator(train_data,steps_per_epoch=steps*8,epochs=10,verbose=2,
                                validation_data=validation_data,validation_steps=steps,initial_epoch=0)
    # to do : save histoty,plot history
    f=h5py.File("Unet-history.h5","w")
    f['dice_coef']=history.history['dice_coef']
    f['val_dice_coef']=history.history['val_dice_coef']
    f.close

    loss,accuracy = model.evaluate_generator(test_data,steps=steps)
    print('\ntest loss',loss)
    print('dice_coef',accuracy)
    #save then delete model
    model.save('Unet-model.h5')
    del model
def tes_generator():
    contains = ['subset0']
    train_data = generate_data_from_file([slices_folder(contain) for contain in contains])
    while 1:
        x, y=train_data.next()
        print x.shape,y.shape
if __name__ == '__main__':
    main()


