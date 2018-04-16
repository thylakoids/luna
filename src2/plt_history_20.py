import h5py
import numpy as np
from matplotlib import pyplot as plt


f10 = h5py.File('Unet-history_10.h5',"r")
f15 = h5py.File('Unet-history_15.h5','r')
f20 = h5py.File('Unet-history_20.h5','r')

train = np.hstack([x['dice_coef'].value for x in [f10,f15,f20]])
vali = np.hstack([x['val_dice_coef'].value for x in [f10,f15,f20]])

plt.plot(train)
plt.plot(vali)
plt.title('model accuracy')
plt.ylabel('dice_coef')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('unet-20.png')