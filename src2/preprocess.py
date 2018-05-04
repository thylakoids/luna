import glob
import gzip
import os
import pickle
import numpy as np
import time

from multiprocessing import cpu_count
from joblib import Parallel, delayed
from skimage import measure,segmentation,morphology
from matplotlib import pyplot as plt
from skimage.filters import roberts
from scipy import ndimage as nd

from utils.visualization import plot_ct_scan
from utils.xyz import load_itk
from utils.pathname import input_folder,mkdir,segmentedLungs_folder,contain_folder

from config import conf

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask2(image, fill_lung_structures=True):

    # not actually binary, but 1 and 2.  1 for lung and air, 2 for others
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    for x in (0, -1):
        for y in (0, -1):
            for z in (0, -1):
                background_label = labels[z, x, y]
                binary_image[background_label == labels] = 2
    half = image.shape[0] / 2
    for pos in ([0, half, half], [-1, half, half],
                [half, 0, half], [half, -1, half],
                [half, half, 0], [half, half, -1]):
        background_label = labels[pos[0], pos[1], pos[2]]
        binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1#0 lung, 1 for others
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0) #air

            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0) #air
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0

    # dialation operation with a ball of radius 10. This operation is
    # to keep nodules attached to the lung wall. However it's very time comsuming
    # so do this dialation when using it as 2D slices
    # binary_image = binary_dialation(binary_image,disk(10))
    return binary_image

def get_segmented_lungs(im, plot=False):
    '''
    This function segments the lungs from the given 2D slice.
    :param im: input gray image
    :param plot:
    :return: binary mask
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < -320
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = segmentation.clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone)
    '''
    Step 3: Label the image.
    '''
    label_image = measure.label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone)
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in measure.regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = morphology.disk(2)
    binary = morphology.binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = morphology.disk(10) #todo : do we need to do this??, or delelet the dilation step when using it
    binary = morphology.binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = nd.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone)
    return binary

def segment_lung_mask1(ct_scan):
    '''
    :param ct_scan: 3D input img
    :return: 3D mask
    '''
    return np.asarray([get_segmented_lungs(slice) for slice in ct_scan])

def processSingle3D(imagePath):
    lung3D,origin,spacing=load_itk(imagePath)
    mask_lung3D = segment_lung_mask1(lung3D)
    meanDiameter= mask_lung3D.sum()**(1.0/3)
    if mask_lung3D.sum() ** (1.0 / 3)>50:
        #save the segmented 3Dlung
        contain = contain_folder(imagePath)
        savePath = imagePath.replace(contain,contain+'_segmentedLungs')
        savePath = savePath.replace('.mhd','.pkl.gz')
        file = gzip.open(savePath,'wb')
        pickle.dump(mask_lung3D,file,protocol=-1)
        pickle.dump(origin,file,protocol=-1)
        pickle.dump(spacing,file,protocol=-1)
        file.close()
        print savePath
    else:
        print '{} can not get good segmentation result'.format(imagePath)

def processfolder(img_folder):
    contain = os.path.split(img_folder)[-1]
    out_folder =  segmentedLungs_folder(contain)
    mkdir(out_folder)
    imagePaths = glob.glob(os.path.join(img_folder,'*.mhd'))
    Parallel(n_jobs=cpu_count()-1)(delayed(processSingle3D)(imagePath) for imagePath in imagePaths)

def test3D():
    start_time = time.time()
    print '{} - start Processing'.format(time.strftime("%H:%M:%S"))
    img_folder = conf.FOLDERS[0]
    imagePaths = glob.glob('{}*.mhd'.format(img_folder))
    for i in range(len(imagePaths)):
        imagePath=imagePaths[i]
        print imagePath
        lung3D,origin,spacing=load_itk(imagePath)
        mask_lung3D = segment_lung_mask1(lung3D)
        plot_ct_scan(lung3D, name='origin{}.png'.format(i), plot=False)
        plot_ct_scan(mask_lung3D, name='mask{}.png'.format(i), plot=False)
    print '{} - Processing took {} seconds'.format(time.strftime("%H:%M:%S"),np.floor(time.time()-start_time))

if __name__ == '__main__':
    if conf.TESTING:
        test3D()
    else:
        start_time = time.time()
        print '{} - start Processing'.format(time.strftime("%H:%M:%S"))
        img_folders = conf.FOLDERS
        for img_folder in img_folders:
            processfolder(input_folder(img_folder))
        print '{} - Processing took {} seconds'.format(time.strftime("%H:%M:%S"),np.floor(time.time()-start_time))


