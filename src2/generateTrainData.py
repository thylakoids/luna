import SimpleITK as sitk
import numpy as np
import os
import pandas as pd
import time
import glob
import skimage.transform
import scipy.ndimage
import pickle
import gzip
import cv2

import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from utils.xyz import load_pickle,load_itk, world_2_voxel, voxel_2_world
from utils.pathname import *
from utils.visualization import plot_ct_scan,plot_3d
from preprocess import normalizePlanes
from scipy import stats


# some constants
annotations_path = '../lunadata/CSVFILE/annotations.csv'
contain = 'rawdata'

mkdir(slicesLungs_folder(contain))
mkdir(slicesNodule_folder(contain))

def draw_circles(image,cands,origin,spacing,size=1):  ##############    to do 
    # image_mask = np.zeros(image.shape)
    # for cand in cands.values:
    #     radius = np.ceil(cand[4])/2
    #     worldCoord = np.asarray([float(cand[3]),float(cand[2]),float(cand[1])])
    #     voxelCoord = world_2_voxel(worldCoord, origin, spacing)
    #     print voxelCoord,radius
    #     for z in range(int(voxelCoord[0]-radius),int(np.ceil(voxelCoord[0]+radius+1))):
    #         radius1=(radius**2-np.absolute(z-voxelCoord[0])**2)
    #         if radius1<=0:
    #             continue
    #         radius1=radius1**0.5
    #         for y in range(int(voxelCoord[1] - radius1), int(np.ceil(voxelCoord[1] + radius1 + 1))):
    #             radius2 = (radius1 ** 2 - np.absolute(y - voxelCoord[1]) ** 2)
    #             if radius2<=0:
    #                 continue
    #             radius2=radius2**0.5
    #             image_mask[z,y,int(voxelCoord[2]-radius2):int(np.ceil(voxelCoord[2]+radius2))]=int(1)
    # return image_mask
    #make empty matrix, which will be filled with the mask
    image_mask = np.zeros(image.shape)
    #run over all the nodules in the lungs
    for ca in cands.values:
        #get middel x-,y-, and z-worldcoordinate of the nodule
        radius = np.ceil(ca[4])/2*size
        coord_x = ca[3]
        coord_y = ca[2]
        coord_z = ca[1]
        image_coord = np.array((coord_x,coord_y,coord_z))

        #determine voxel coordinate given the worldcoordinate
        image_coord = world_2_voxel(image_coord,origin,spacing)

        #determine the range of the nodule
        noduleRange = np.linspace(-radius, radius,ca[4]*4*size)

        #create the mask
        for x in noduleRange:
            for y in noduleRange:
                for z in noduleRange:
                    coords = world_2_voxel(np.array((coord_x+x,coord_y+y,coord_z+z)),origin,spacing)
                    dis=np.linalg.norm(image_coord-coords)
                    if dis<= radius:
                        if size>1:
                            image_mask[int(np.round(coords[0])),int(np.round(coords[1])),int(np.round(coords[2]))] = stats.norm.pdf(dis,0,radius/size)
                        else:
                            image_mask[int(np.round(coords[0])),int(np.round(coords[1])),int(np.round(coords[2]))] = int(1)
    image_mask=image_mask/image_mask.max()
    return image_mask

def show_slice(imagePath, annotations):
    image, origin, spacing = load_pickle(imagePath)
    image=normalizePlanes(image)
    image[image==0]=1 #background bright
    #determine the annotations in a lung from csv file
    imageName = os.path.split(imagePath)[1].replace('.pkl.gz','')
    print imageName
    image_annotations = annotations[annotations['seriesuid'] == imageName]
    nodule_mask = draw_circles(image,image_annotations,origin,spacing,size=2)
    # plot_3d(image)
    # plot_ct_scan(image)
    # plot_ct_scan(nodule_mask)

    for ca in image_annotations.values:
        #get middel x-,y-, and z-worldcoordinate of the nodule
        radius = np.ceil(ca[4])/2
        coord_x = ca[3]
        coord_y = ca[2]
        coord_z = ca[1]
        image_coord = np.array((coord_x,coord_y,coord_z))
        #determine voxel coordinate given the worldcoordinate
        image_coord = world_2_voxel(image_coord,origin,spacing)
        print image_coord,radius


        # patchWidth=50
        # xs=int(image_coord[1]-patchWidth)
        # xe=int(image_coord[1]+patchWidth)
        # ys=int(image_coord[2]-patchWidth)
        # ye=int(image_coord[2]+patchWidth)
        z=int(image_coord[0])

        # plt.figure()
        # plt.imshow(image[z])
        # plt.figure()
        # plt.imshow(nodule_mask[z],cmap=plt.cm.bone)
        # plt.show()
        
        
        
        # nodule_s=nodule_mask[z,xs:xe,ys:ye]
        # image_s=image[z,xs:xe,ys:ye]
        nodule_s=nodule_mask[z,:,:]
        image_s=image[z,:,:]
        # image_3c=np.stack((image_s,image_s,image_s)).transpose(1,2,0)
        # plt.figure(1)
        # plt.imshow(nodule_s,cmap=plt.cm.bone)
        # plt.subplot(221)
        # plt.subplot(222)
        # plt.imshow(image_s,cmap=plt.cm.gray)
        # plt.subplot(223)
        # plt.imshow(image_3c) 

        
        image_s_nodule=image_s.copy()
        image_s_lung=image_s.copy()
        image_s_nodule[nodule_s==0]=0
        image_s_lung[nodule_s!=0]=0

        image_3c=np.stack((image_s_lung,image_s_lung,image_s_lung)).transpose(1,2,0)
        image_nodule_3c=np.stack((image_s_nodule,image_s_nodule,image_s_nodule)).transpose(1,2,0)
        alpha=0.6
        col=np.array([222,129,0])/255.0
        col = np.array([175,99,37])/255.0/alpha
        nodule_3c=np.stack((nodule_s*col[0],nodule_s*col[1],nodule_s*col[2])).transpose(1,2,0)
        

        plt.figure(2)
        plt.subplot(221)
        plt.imshow(nodule_3c)
        plt.subplot(222)
        plt.imshow(image_s_lung,cmap=plt.cm.bone)
        plt.subplot(223)
        plt.imshow(image_3c+(1-alpha)*image_nodule_3c+alpha*nodule_3c)
        plt.subplot(224)
        plt.imshow((1-alpha)*image_3c+(1-alpha)*image_nodule_3c+alpha*nodule_3c)
        plt.show()
def minmax(array):
    return array.min(),array.max()
def create_slice(imagePath,annotations):
    image, origin, spacing = load_pickle(imagePath)
    # image = normalizePlanes(image) # normalized online to save storage space?
    image[image<-1000]=-1000 #where has lung?
    # determine the annotations in a lung from csv file
    imageName = os.path.split(imagePath)[1].replace('.pkl.gz', '')
    print imageName
    image_annotations = annotations[annotations['seriesuid'] == imageName]


    # calculate resize factor
    resize_shape = image.shape*spacing
    new_shape = np.round(resize_shape)
    new_resize = new_shape / image.shape
    new_spacing = spacing /new_resize

    #resize image & resize nodule_mask with bilinear interpolation,still int16
    resize_image = scipy.ndimage.zoom(image,new_resize,order=1)
    nodule_mask = draw_circles(resize_image, image_annotations, origin, new_spacing, size=1)

    #Determine which slices contain nodules
    #todo: 1.positive slices and negative slices 2. in z,y,x direction
    maxshape=0
    for z in range(nodule_mask.shape[0]):
        img = resize_image[z]
        mask = nodule_mask[z]
        try:
            # x: axis 1,y:axis0
            x_min,x_max = minmax(np.where(img.sum(axis=0)>-1000*img.shape[1])[0])
            y_min,y_max = minmax(np.where(img.sum(axis=1)>-1000*img.shape[0])[0])
            imgSub = img[y_min:y_max, x_min:x_max]
            maxshape_now = max(imgSub.shape)
            if maxshape_now > maxshape:
                maxshape = maxshape_now
        except ValueError:
            print 'no lung pixel detected'
    print maxshape
    #save slices

    #todo 1.determine lung region 2.padding to ?? then resize to 256,to make sure the spacing is the same










if __name__=='__main__':
    annotations = pd.read_csv(annotations_path)
    contain = 'rawdata'
    segmentedLungsPaths=segmentedLungs_folder(contain)
    imagePaths = glob.glob('{}*.pkl.gz'.format(segmentedLungsPaths))
    # show_slice(imagePaths[3],annotations)

    # create_slice(imagePaths[3],annotations)
    img,_,_=load_pickle(imagePaths[3]) # todo: how to segment
    plot_ct_scan(img)



