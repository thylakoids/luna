import numpy as np
import SimpleITK as sitk
import gzip
import pickle

from skimage.morphology import disk,binary_dilation,binary_closing
from matplotlib import pyplot as plt

def load_itk(filename):
    itkimage = sitk.ReadImage(filename)
    image = sitk.GetArrayFromImage(itkimage)
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    return image, origin, spacing # image.shape=(126,512,512)
def world_2_voxel(world_coord, origin, spacing):
    stretched_voxel_coord = np.absolute(world_coord - origin)
    voxel_coord = stretched_voxel_coord / spacing
    return voxel_coord

def voxel_2_world(voxel_coord, origin, spacing):
    stretched_voxel_coord = voxel_coord * spacing
    world_coord = stretched_voxel_coord + origin
    return world_coord
def load_pickle(filename):
    file = gzip.open(filename,'rb')
    image = pickle.load(file)
    origin = pickle.load(file)
    spacing = pickle.load(file)
    file.close()
    return image, origin, spacing
def load_slice(filename):
    file = gzip.open(filename,'rb')
    image = pickle.load(file) #int 16
    lung_mask= pickle.load(file) #int 8
    nodule_mask= pickle.load(file)#int 8
    origin = pickle.load(file)
    spacing = pickle.load(file)
    file.close()
    # print lung_mask.dtype
    lung_mask = binary_dilation(lung_mask,disk(2))
    lung_mask = binary_closing(lung_mask,disk(10))
    return image, lung_mask, nodule_mask, origin, spacing
if __name__ == "__main__":
    image, origin, spacing = load_itk('../../lunadata/rawdata/1.3.6.1.4.1.14519.5.2.1.6279.6001.317087518531899043292346860596.mhd')
    print 'slice0:\n',image[:,:,0]
    print 'Shape:', image.shape
    print 'Origin:', origin
    print 'Spacing:', spacing

    plt.hist(image.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()

    # Show some slice in the middle
    plt.imshow(image[240], cmap=plt.cm.gray)
    plt.colorbar()
    plt.show()

