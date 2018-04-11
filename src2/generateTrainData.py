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
from typing import Optional

from utils.xyz import load_pickle, load_itk, load_slice, world_2_voxel, voxel_2_world
from utils.pathname import *
from utils.visualization import plot_ct_scan, plot_3d
from utils.normalize import normalizePlanes
from scipy import stats
from skimage.morphology import disk,binary_dilation,binary_closing

# some constants
annotations_path = '../lunadata/CSVFILE/annotations.csv'
contain = 'rawdata'
mkdir_iter(slices_folder(contain))


def draw_circlesV2(image, cands, origin, spacing):
    # looks more clever, but less precise than draw_circles
    # make empty matrix, which will be filled with the mask
    image_mask = np.zeros(image.shape)
    # run over all the nodules in the lungs
    for ca in cands.values:
        radius = ca[4] / 2
        world_ball = ball(radius)
        image_ball = scipy.ndimage.zoom(world_ball, 1 / spacing)
        world_origin = np.array((ca[3], ca[2], ca[1]))
        image_origin = world_2_voxel(world_origin, origin, spacing)  # origin
        ball_origin = [int(np.round(image_origin[0] - image_ball.shape[0]/2)),
                       int(np.round(image_origin[1] - image_ball.shape[1]/2)),
                       int(np.round(image_origin[2] - image_ball.shape[2]/2))]
        # insert the ball into the mask
        image_mask[ball_origin[0]:(ball_origin[0] + image_ball.shape[0]),
        ball_origin[1]:(ball_origin[1] + image_ball.shape[1]),
        ball_origin[2]:(ball_origin[2] + image_ball.shape[2])]=image_ball
    return image_mask


def draw_circles(image, cands, origin, spacing, size=1):
    # make empty matrix, which will be filled with the mask
    image_mask = np.zeros(image.shape)

    # run over all the nodules in the lungs
    for ca in cands.values:
        # get middel x-,y-, and z-worldcoordinate of the nodule
        radius = ca[4] / 2
        coord_x = ca[3]
        coord_y = ca[2]
        coord_z = ca[1]
        # determine the range of the nodule
        noduleRange = np.linspace(-radius*size, radius*size, np.ceil(ca[4] * 2 * size))

        # create the mask
        for x in noduleRange:
            for y in noduleRange:
                for z in noduleRange:
                    dis = np.linalg.norm([x,y,z])
                    if dis <= radius*size:
                        coords = world_2_voxel(
                            np.array((coord_x + x, coord_y + y, coord_z + z)),
                            origin, spacing)
                        if size > 1:
                            image_mask[int(np.round(coords[0])), int(
                                np.round(coords[1])), int(
                                np.round(coords[2]))] = stats.norm.pdf(dis/radius)
                        else:
                            image_mask[int(np.round(coords[0])), int(
                                np.round(coords[1])), int(
                                np.round(coords[2]))] = int(1)
    image_mask = image_mask / image_mask.max()
    if size>1:
        return image_mask
    else:
        return image_mask.astype('int8')


def show_circle(imagePath, annotations):
    lung_mask, origin, spacing = load_pickle(imagePath)
    # image = normalizePlanes(image) # normalized online to save storage space?
    # determine the annotations in a lung from csv file
    imageName = os.path.split(imagePath)[1].replace('.pkl.gz', '')
    lung,_,_=load_itk(input_folder(contain)+imageName+'.mhd')

    # determine the annotations in a lung from csv file
    imageName = os.path.split(imagePath)[1].replace('.pkl.gz', '')
    print imageName
    image_annotations = annotations[annotations['seriesuid'] == imageName]
    nodule_mask = draw_circles(lung_mask, image_annotations, origin, spacing,size=2)
    for ca in image_annotations.values:
        # get middel x-,y-, and z-worldcoordinate of the nodule
        radius = np.ceil(ca[4]) / 2
        image_coord = np.array((ca[3], ca[2], ca[1]))
        image_coord = world_2_voxel(image_coord, origin, spacing)
        print image_coord, radius
        z = int(image_coord[0])
        nodule_s = nodule_mask[z]
        image_s = lung[z]
        image_s = normalizePlanes(image_s)

        mask_s = binary_dilation(lung_mask[z],disk(2))
        mask_s = binary_closing(mask_s,disk(10))
        image_s[mask_s==0]= 1 #make it bright



        image_s_nodule = image_s.copy()
        image_s_lung = image_s.copy()
        image_s_nodule[nodule_s == 0] = 0
        image_s_lung[nodule_s != 0] = 0

        image_3c = np.stack(
            (image_s_lung, image_s_lung, image_s_lung)).transpose(1, 2, 0)
        image_nodule_3c = np.stack(
            (image_s_nodule, image_s_nodule, image_s_nodule)).transpose(1, 2, 0)
        alpha = 0.6
        col = np.array([222, 129, 0]) / 255.0
        # col = np.array([175, 99, 37]) / 255.0 / alpha
        nodule_3c = np.stack((nodule_s * col[0], nodule_s * col[1],
                              nodule_s * col[2])).transpose(1, 2, 0)

        plt.figure()
        plt.subplot(221)
        plt.imshow(mask_s,cmap=plt.cm.bone)
        plt.subplot(222)
        plt.imshow(image_s, cmap=plt.cm.bone)
        plt.subplot(223)
        plt.imshow(image_3c + (1 - alpha) * image_nodule_3c + alpha * nodule_3c)
        plt.subplot(224)
        plt.imshow((1 - alpha) * image_3c + (
                    1 - alpha) * image_nodule_3c + alpha * nodule_3c)
    plt.show()


def minmax(array):
    return array.min(), array.max()


def create_slice(imagePath, annotations):
    lung_mask, origin, spacing = load_pickle(imagePath)
    # image = normalizePlanes(image) # normalized online to save storage space?
    # determine the annotations in a lung from csv file
    imageName = os.path.split(imagePath)[1].replace('.pkl.gz', '')
    lung,_,_=load_itk(input_folder(contain)+imageName+'.mhd')
    print imageName
    image_annotations = annotations[annotations['seriesuid'] == imageName]

    # calculate resize factor
    resize_shape = lung.shape * spacing
    new_shape = np.round(resize_shape)
    new_resize = new_shape / lung.shape
    new_spacing = spacing / new_resize

    # resize image & resize nodule_mask with bilinear interpolation,still int16
    resize_lung = scipy.ndimage.zoom(lung, new_resize, order=1)
    resize_lung_mask = scipy.ndimage.zoom(lung_mask, new_resize, order=1)
    assert(spacing[1]==spacing[2],'x spacing != y spacing')
    # padding to 400, adjust the size, find the maximum of resize_image.shape[1] todo
    padding_shape=400
    assert(padding_shape>=resize_shape[1],'padding size < resize image shape')
    assert(new_shape==resize_lung.shape,'????')


    padding_lung = np.zeros((resize_lung.shape[0],padding_shape,padding_shape),dtype='int16')-3000
    padding_lung_mask= np.zeros((resize_lung_mask.shape[0],padding_shape,padding_shape),dtype='int8')
    offset = padding_shape - resize_lung.shape[1]
    upper_offset = np.round(offset / 2)
    lower_offset = offset - upper_offset

    new_origin = voxel_2_world([0, -upper_offset, -upper_offset], origin, new_spacing)
    padding_lung[:,upper_offset:-lower_offset,upper_offset:-lower_offset] = resize_lung
    padding_lung_mask[:, upper_offset:-lower_offset, upper_offset:-lower_offset] = resize_lung_mask


    nodule_mask = draw_circles(padding_lung, image_annotations, new_origin, new_spacing)

    for z in range(nodule_mask.shape[0]):
        lung = padding_lung[z]
        lung_mask = padding_lung_mask[z]
        mask = nodule_mask[z]
        if lung_mask.sum()>0:#this slice has some lung
            if mask.sum()>0:#this slice has nodule
                savePath = '{}{}_Slice{}.+z.pkl.gz'.format(slices_folder(contain),imageName,z)
            else:#this slice has lung but no nodule
                savePath = '{}{}_Slice{}.-z.pkl.gz'.format(slices_folder(contain), imageName, z)
            file = gzip.open(savePath, 'wb')
            pickle.dump(lung, file, protocol=-1)
            pickle.dump(lung_mask,file,protocol=-1)
            pickle.dump(mask, file, protocol=-1)
            pickle.dump(new_origin, file, protocol=-1)
            pickle.dump(new_spacing,file,protocol=-1)
            file.close()
    for y in range(nodule_mask.shape[1]):
        lung = padding_lung[:,y,:]
        lung_mask = padding_lung_mask[:,y,:]
        mask = nodule_mask[:,y,:]
        if lung_mask.sum()>0:#this slice has some lung
            if mask.sum()>0:#this slice has nodule
                savePath = '{}{}_Slice{}.+y.pkl.gz'.format(slices_folder(contain),imageName,y)
            else:#this slice has lung but no nodule
                savePath = '{}{}_Slice{}.-y.pkl.gz'.format(slices_folder(contain),imageName,y)
            file = gzip.open(savePath, 'wb')
            pickle.dump(lung, file, protocol=-1)
            pickle.dump(lung_mask, file, protocol=-1)
            pickle.dump(mask,file,protocol=-1)
            pickle.dump(new_origin, file, protocol=-1)
            pickle.dump(new_spacing,file,protocol=-1)
            file.close()
    for x in range(nodule_mask.shape[2]):
        lung = padding_lung[:,:,x]
        lung_mask = padding_lung_mask[:,:,x]
        mask = nodule_mask[:,:,x]
        if lung_mask.sum()>0:#this slice has some lung
            if mask.sum()>0:#this slice has nodule
                savePath = '{}{}_Slice{}.+x.pkl.gz'.format(slices_folder(contain),imageName,x)
            else:#this slice has lung but no nodule
                savePath = '{}{}_Slice{}.-x.pkl.gz'.format(slices_folder(contain),imageName,x)
            file = gzip.open(savePath, 'wb')
            pickle.dump(lung, file, protocol=-1)
            pickle.dump(lung_mask, file, protocol=-1)
            pickle.dump(mask,file,protocol=-1)
            pickle.dump(new_origin, file, protocol=-1)
            pickle.dump(new_spacing,file,protocol=-1)
            file.close()

def show_slice(num=10):
    imagePaths = glob.glob('{}*{}.pkl.gz'.format(slices_folder(contain),'+x'))
    imagePaths = imagePaths[:num]
    for imagePath in imagePaths:
        img, lung_mask, nodule_mask,_,_=load_slice(imagePath)
        plt.subplot(221)
        plt.imshow(img,cmap=plt.cm.gray)
        plt.subplot(222)
        plt.imshow(lung_mask,cmap=plt.cm.gray)
        plt.subplot(223)
        plt.imshow(nodule_mask,cmap=plt.cm.gray)
        plt.show()


if __name__ == '__main__':
    annotations = pd.read_csv(annotations_path)
    contain = 'rawdata'
    segmentedLungsPaths = segmentedLungs_folder(contain)
    imagePaths = glob.glob('{}*.pkl.gz'.format(segmentedLungsPaths))
    # show_circle(imagePaths[2],annotations)
    # create_slice(imagePaths[2],annotations)
    show_slice(10)
    #todo load_data()

