from keras.models import load_model
from loadDataTrain import generate_data_from_file
from UNET import dice_coef,dice_coef_loss


import glob
import random
import cv2
import skvideo.io
import re

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from generateTrainData import draw_circles2
from utils.pathname import *
from utils.normalize import normalizePlanes
from utils.visualization import impose2
from utils.xyz import load_slice, load_itk
from config import conf
def vispre():
    #test data
    contains_test = conf.FOLDERS
    test_data = generate_data_from_file([slices_folder(contain) for contain in contains_test])
    #load model
    model = load_model("Unet-model.h5", custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})


    X,Y = test_data.next()
    print X.shape
    Y_pre = model.predict(X,verbose=0)
    _,plots = plt.subplots(3,4)
    for i in range(4):
        plots[0,i].imshow(X[i][0],cmap='gray')
        plots[1,i].imshow(Y[i][0],cmap='gray')
        plots[2,i].imshow(Y_pre[i][0],cmap='gray')
    plt.show()
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
def visImpose(contain='rawdata',num=10):
    imagePaths = glob.glob(os.path.join(slices_folder(contain), '*+z.pkl.gz'))
    random.shuffle(imagePaths)
    imagePaths = imagePaths[:num]
    for imagePath in imagePaths:
        img, lung_mask, nodule_mask, _, _ = load_slice(imagePath)
        img = normalizePlanes(img)
        img[lung_mask==0]=1 # bright
        img_impose = impose(img,nodule_mask)
        plt.imshow(img_impose)
        plt.show()
def Savevideo():
    #path
    mkdir('../etc')
    img_folder = conf.FOLDERS[0]
    path = os.path.join(input_folder(img_folder),'*.mhd')
    imagePaths = glob.glob(path)
    for imagePath in imagePaths:
        # imagePath=imagePaths[2]
        #annotations
        annotations = pd.read_csv(conf.ANNOTATION_PATH)
        imageName = os.path.basename(imagePath).replace('.mhd', '')
        image_annotations = annotations[annotations['seriesuid'] == imageName]
        #lung
        lung3D, origin, spacing = load_itk(imagePath)
        lung3D = normalizePlanes(lung3D)
        #mask
        nodule_mask = draw_circles2(lung3D, image_annotations, origin, spacing, size=4)
        #impose and show
        # img = impose2(lung3D, nodule_mask)
        # plt.imshow(img[214])#70 for 0
        # plt.show()

        #video
        img = impose2(lung3D, nodule_mask)
        img = img*255
        img = img.astype(np.uint8)


        rate='10'
        inputdict={
        '-r':rate,
        }
        outputdict={
        '-vcodec': 'libx264',
        '-pix_fmt': 'yuv420p',
        '-b':'300000000',
        '-r': rate,
        }
        skvideo.io.vwrite('../etc/{}.mp4'.format(imageName),img,inputdict=inputdict,outputdict=outputdict)
        print imageName
def video2jpg(path):
    outdir = re.findall('(^.+)(?:.mp4|.avi)$',path)[0]
    mkdir_iter(outdir)
    vid = skvideo.io.vread(path)
    for i in range(vid.shape[0]):
        outpath = os.path.join(outdir,'{}.jpg'.format(i))
        skvideo.io.vwrite(outpath,vid[i])
if __name__ =='__main__':
    vispre()
    # Savevideo()
    # video2jpg('../etc/1.3.6.1.4.1.14519.5.2.1.6279.6001.249530219848512542668813996730.mp4')

