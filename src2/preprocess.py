import numpy as np
import os
import time 
import glob
import pickle
import gzip


from utils.visualization import plot_ct_scan,plot_3d
from utils.xyz import load_itk
from joblib import Parallel, delayed
from matplotlib import pyplot as plt 
from multiprocessing import cpu_count

from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi





#Some constants
PIXEL_MEAN=0.11 ######### 0 for not do zero centering
nCPU=cpu_count()




def zero_center(image):
    image = image - PIXEL_MEAN
    return image
def normalizePlanes(npzarray):
     
    maxHU = 400.
    minHU = -1000.
 
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray
def get_segmented_lungs(img):
    # Step 1: Convert into a binary image. 
    binary_image = np.array(img < -320)
    # Step 2: remove the blobs in the corners
    label_image=measure.label(binary_image)
    for i,j in zip([0,0,-1,-1],[0,-1,0,-1]):
        background_label=label_image[i,j]
        binary_image[label_image==background_label]=0
    # Step 3: Erosion operation with a disk of radius 2. This operation is seperate the lung nodules attached to the blood vessels.
    selem = disk(2)
    erosion_image = binary_erosion(binary_image, selem)
    # Step 4: Closure operation with a disk of radius 10. This operation is to keep nodules attached to the lung wall.
    selem = disk(10)
    closing_image = binary_closing(erosion_image, selem)
    # Step 5: Fill in the small holes inside the binary mask of lungs.
    edges = roberts(closing_image)
    binary = ndi.binary_fill_holes(edges)
    # Step 6: Superimpose the binary mask on the input image.
    img[binary==0]=-1000
    # Step 7: normalized to (0,1)
    normalize_image=normalizePlanes(img)
    # Step 8: zero center
    zero_image=zero_center(normalize_image)
    return zero_image
def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
    return binary_image
def processSingle3D(imagePath):
    lung3D,origin,spacing=load_itk(imagePath)
    mask_lung3D = segment_lung_mask(lung3D, True)
    lung3D[mask_lung3D==0]=-1000 #0 water
    #save the segmented 3Dlung
    contain = imagePath.split('/')[-2]
    savePath = imagePath.replace(contain,contain+'_segmentedLungs')
    savePath = savePath.replace('.mhd','.pkl.gz')
    file = gzip.open(savePath,'wb')
    pickle.dump(lung3D,file,protocol=-1)
    pickle.dump(origin,file,protocol=-1)
    pickle.dump(spacing,file,protocol=-1)
    file.close()
    print savePath

def processSingle2D(imagePath):



    lung3D,origin,spacing=load_itk(imagePath)
    segmented_lung2D_list=[get_segmented_lungs(slice) for slice in lung3D]

    segmented_lung3D=np.stack(segmented_lung2D_list)
    #save the segmented 3Dlung
    contain = imagePath.split('/')[-2]
    savePath = imagePath.replace(contain,contain+'_segmentedLungs')
    savePath = savePath.replace('.mhd','.pkl.gz')
    file = gzip.open(savePath,'wb')
    pickle.dump(segmented_lung3D,file,protocol=-1)
    pickle.dump(origin,file,protocol=-1)
    pickle.dump(spacing,file,protocol=-1)
    file.close()
    print savePath
def processfolder(img_folder):
    contain=img_folder.split('/')[-2]
    out_folder =  img_folder.replace(contain,contain+'_segmentedLungs/')
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    imagePaths = glob.glob('{}*.mhd'.format(img_folder))    
    Parallel(n_jobs=nCPU)(delayed(processSingle3D)(imagePath) for imagePath in imagePaths)

def test3D():
    start_time = time.time()
    print '{} - start Processing'.format(time.strftime("%H:%M:%s"))
    img_folder = '../lunadata/rawdata/'
    # processfolder(img_folder)
    imagePaths = glob.glob('{}*.mhd'.format(img_folder)) 
    imagePath=imagePaths[2]
    lung3D,origin,spacing=load_itk(imagePath)
    mask_lung3D = segment_lung_mask(lung3D, True)
    lung3D[mask_lung3D==0]=0 #0 water
    plot_3d(lung3D,-320,0.25)
    print '{} - Processing took {} seconds'.format(time.strftime("%H:%M:%s"),np.floor(time.time()-start_time))

if __name__ == '__main__':
    start_time = time.time()
    print '{} - start Processing'.format(time.strftime("%H:%M:%s"))
    img_folder = '../lunadata/rawdata/'
    processfolder(img_folder)
    print '{} - Processing took {} seconds'.format(time.strftime("%H:%M:%s"),np.floor(time.time()-start_time))

    # test3D()

