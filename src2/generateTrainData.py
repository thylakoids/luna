import glob
import gzip
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as nd
from scipy import stats
from skimage.morphology import ball
from multiprocessing import cpu_count
from joblib import Parallel, delayed

from utils.pathname import *
from utils.xyz import load_pickle, load_itk, load_slice, world_2_voxel, voxel_2_world
from utils.visualization import myball

from config import conf



def draw_circles2(image, cands, origin, spacing, size=1):
    # looks more clever, but less precise than draw_circles
    # make empty matrix, which will be filled with the mask
    image_mask = np.zeros(image.shape)
    # run over all the nodules in the lungs
    for ca in cands.values:
        radius = ca[4] / 2
        world_ball = myball(radius,size)
        image_ball = nd.interpolation.zoom(world_ball, 1 / spacing,order=0)
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
    if size>1:
        image_mask = image_mask / image_mask.max()
        return image_mask
    else:
        return image_mask.astype('int8')
def create_slice(imagePath, annotations, contain):
    print imagePath
    lung_mask, origin, spacing = load_pickle(imagePath)
    # determine the annotations in a lung from csv file
    imageName = os.path.basename(imagePath).replace('.pkl.gz', '')
    lung,_,_=load_itk(os.path.join(input_folder(contain),imageName+'.mhd'))
    image_annotations = annotations[annotations['seriesuid'] == imageName]

    # calculate resize factor
    resize_shape = lung.shape * spacing
    new_shape = np.round(resize_shape)
    new_resize = new_shape / lung.shape
    new_spacing = spacing / new_resize

    # resize image & resize nodule_mask with nearest neighbor interpolation,still int16?
    resize_lung = nd.interpolation.zoom(lung, new_resize, order=0)
    resize_lung_mask = nd.interpolation.zoom(lung_mask, new_resize, order=0)
    assert(spacing[1]==spacing[2],'x spacing != y spacing')
    padding_shape=512
    assert(padding_shape>=resize_shape[1],'padding size < resize image shape')
    assert(new_shape==resize_lung.shape,'????')


    padding_lung = np.zeros((resize_lung.shape[0],padding_shape,padding_shape),dtype='int16')-3000
    padding_lung_mask= np.zeros((resize_lung_mask.shape[0],padding_shape,padding_shape),dtype='int8')
    offset = padding_shape - resize_lung.shape[1]
    upper_offset = np.round(offset / 2)
    lower_offset = offset - upper_offset

    new_origin = voxel_2_world([0, -upper_offset, -upper_offset], origin, new_spacing)
    if offset>0:
        padding_lung[:,upper_offset:-lower_offset,upper_offset:-lower_offset] = resize_lung
        padding_lung_mask[:, upper_offset:-lower_offset, upper_offset:-lower_offset] = resize_lung_mask
    else:
        padding_lung=resize_lung#offset==0
        padding_lung_mask=resize_lung_mask


    nodule_mask = draw_circles(padding_lung, image_annotations, new_origin, new_spacing)

    for z in range(nodule_mask.shape[0]):
        lung = padding_lung[z]
        lung_mask = padding_lung_mask[z]
        mask = nodule_mask[z]
        if lung_mask.sum()>0:#this slice has some lung
            if mask.sum()>0:#this slice has nodule
                savePath = os.path.join(slices_folder(contain),'{}_slice{}.+z.pkl.gz'.format(imageName,z))
            else:#this slice has lung but no nodule
                savePath = os.path.join(slices_folder(contain), '{}_slice{}.-z.pkl.gz'.format(imageName, z))
            file = gzip.open(savePath, 'wb')
            pickle.dump(lung, file, protocol=-1)
            pickle.dump(lung_mask,file,protocol=-1)
            pickle.dump(mask, file, protocol=-1)
            pickle.dump(new_origin, file, protocol=-1)
            pickle.dump(new_spacing,file,protocol=-1)
            file.close()
    # for y in range(nodule_mask.shape[1]):
    #     lung = padding_lung[:,y,:]
    #     lung_mask = padding_lung_mask[:,y,:]
    #     mask = nodule_mask[:,y,:]
    #     if lung_mask.sum()>0:#this slice has some lung
    #         if mask.sum()>0:#this slice has nodule
    #             savePath = '{}{}_Slice{}.+y.pkl.gz'.format(slices_folder(contain),imageName,y)
    #         else:#this slice has lung but no nodule
    #             savePath = '{}{}_Slice{}.-y.pkl.gz'.format(slices_folder(contain),imageName,y)
    #         file = gzip.open(savePath, 'wb')
    #         pickle.dump(lung, file, protocol=-1)
    #         pickle.dump(lung_mask, file, protocol=-1)
    #         pickle.dump(mask,file,protocol=-1)
    #         pickle.dump(new_origin, file, protocol=-1)
    #         pickle.dump(new_spacing,file,protocol=-1)
    #         file.close()
    # for x in range(nodule_mask.shape[2]):
    #     lung = padding_lung[:,:,x]
    #     lung_mask = padding_lung_mask[:,:,x]
    #     mask = nodule_mask[:,:,x]
    #     if lung_mask.sum()>0:#this slice has some lung
    #         if mask.sum()>0:#this slice has nodule
    #             savePath = '{}{}_Slice{}.+x.pkl.gz'.format(slices_folder(contain),imageName,x)
    #         else:#this slice has lung but no nodule
    #             savePath = '{}{}_Slice{}.-x.pkl.gz'.format(slices_folder(contain),imageName,x)
    #         file = gzip.open(savePath, 'wb')
    #         pickle.dump(lung, file, protocol=-1)
    #         pickle.dump(lung_mask, file, protocol=-1)
    #         pickle.dump(mask,file,protocol=-1)
    #         pickle.dump(new_origin, file, protocol=-1)
    #         pickle.dump(new_spacing,file,protocol=-1)
    #         file.close()

def tes_realLungSize():
    contains = ['subset{}'.format(i) for i in range(10)]
    real_shapes=[]
    for contain in contains:
        print contain
        img_folder = input_folder(contain)
        imagePaths = glob.glob('{}*.mhd'.format(img_folder))
        for imagePath in imagePaths:
            print imagePath
            lung3D, _, spacing = load_itk(imagePath)
            real_shape=lung3D.shape*spacing
            real_shapes.append(real_shape)
    result = np.array(real_shapes)
    return result # array([165.5, 236. , 236. ]) array([416., 499.99975586, 499.99975586])
if __name__ == '__main__':
    annotations_path = conf.ANNOTATION_PATH
    annotations = pd.read_csv(annotations_path)
    contains = conf.FOLDERS
    for contain in contains:
        mkdir_iter(slices_folder(contain))
        segmentedLungsPaths = segmentedLungs_folder(contain)
        imagePaths = glob.glob(os.path.join(segmentedLungsPaths,'*.pkl.gz'))
        if conf.TESTING:
            for imagePath in imagePaths:
                print imagePath
                create_slice(imagePath, annotations,contain)
                break
        else:
            Parallel(n_jobs=cpu_count()-1)(delayed(create_slice)(imagePath, annotations,contain) for imagePath in imagePaths)
