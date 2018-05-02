# from keras.models import load_model
# from loadDataTrain import generate_data_from_file
# from UNET import dice_coef,dice_coef_loss
from utils.pathname import *
import matplotlib.pyplot as plt
import glob
from utils.xyz import load_slice
import random
import numpy as np
from utils.normalize import normalizePlanes
def vispre():
    #test data
    contains_test = ['subset{}'.format(9)]
    test_data = generate_data_from_file([slices_folder(contain) for contain in contains_test])
    #load model
    model = load_model("Unet-model.h5", custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})


    X,Y = test_data.next()
    Y_pre = model.predict(X,verbose=0)
    _,plots = plt.subplots(3,4)
    for i in range(X.shape[0]):
        plots[0,i].imshow(X[i][0])
        plots[1,i].imshow(Y[i][0])
        plots[2,i].imshow(Y_pre[i][0])
    plt.savefig('predict.png')
def visSlice(contain='rawdata',num=10):
    imagePaths = glob.glob(os.path.join(slices_folder(contain),'*+z.pkl.gz'))
    random.shuffle(imagePaths)
    imagePaths = imagePaths[:num]
    for imagePath in imagePaths:
        img, lung_mask, nodule_mask, _, _ = load_slice(imagePath)
        plt.subplot(221)
        plt.imshow(img, cmap=plt.cm.gray)
        plt.subplot(222)
        plt.imshow(lung_mask, cmap=plt.cm.gray)
        plt.subplot(223)
        plt.imshow(nodule_mask, cmap=plt.cm.gray)
        plt.savefig(os.path.basename(imagePath).replace('.pkl.gz','.png'))
def impose(img,mask,col = np.array([256, 0, 0])/ 255.0,alpha=0.5):
    img_lung = img.copy()
    img_nodule = img.copy()
    img_lung[mask==1]=0
    img_nodule[mask==0]=0

    img_nodule_3c = np.stack(
        (img_nodule, img_nodule, img_nodule)).transpose(1, 2, 0)
    img_lung_3c = np.stack(
        (img_lung, img_lung, img_lung)).transpose(1, 2, 0)
    mask_3c = np.stack(
        (mask * col[0], mask * col[1],mask * col[2])).transpose(1, 2, 0)



    img_impose = img_lung_3c+(1-alpha)*img_nodule_3c+alpha*mask_3c
    plt.imshow(img_impose)
    plt.show()
def visImpose(contain='rawdata',num=10):
    imagePaths = glob.glob(os.path.join(slices_folder(contain), '*+z.pkl.gz'))
    random.shuffle(imagePaths)
    imagePaths = imagePaths[:num]
    for imagePath in imagePaths:
        img, lung_mask, nodule_mask, _, _ = load_slice(imagePath)
        img = normalizePlanes(img)
        img[lung_mask==0]=1 # bright
        impose(img,nodule_mask)

if __name__ =='__main__':
    visImpose()

