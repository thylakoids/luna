#%%
from scipy import ndimage as nd
from skimage.transform import resize,rescale
from  matplotlib import pyplot as plt
import random
import numpy as np

from utils.pathname import *
import glob
from utils.xyz import load_slice
#%%
def visSlice(contain='rawdata',num=10):
    imagePaths = glob.glob(os.path.join(slices_folder(contain),'*+z.pkl.gz'))
    random.shuffle(imagePaths)
    imagePaths = imagePaths[:num]
    for imagePath in imagePaths:
        img, lung_mask, nodule_mask, _, _ = load_slice(imagePath)
        plt.subplot(221)
        plt.imshow(lung_mask, cmap=plt.cm.gray)
        plt.title('origin')
        plt.subplot(222)
        plt.imshow(nd.interpolation.zoom(lung_mask,2), cmap=plt.cm.gray)
        plt.title('nd.interpolation.zoom')
        plt.subplot(223)
        plt.imshow(nd.zoom(lung_mask,0.5), cmap=plt.cm.gray)
        plt.title('nd.zoom')
        plt.subplot(224)
        plt.imshow(resize(lung_mask,np.array(lung_mask.shape)*0.5), cmap=plt.cm.gray)
        plt.title('resize')
        plt.show()
        break

if __name__ == '__main__':
    visSlice()