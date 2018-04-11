import glob
import gzip
import os
import pickle
import numpy as np
import time

from multiprocessing import cpu_count
from joblib import Parallel, delayed
from skimage import measure
from skimage.morphology import binary_closing,ball,binary_dilation

from utils.visualization import plot_ct_scan
from utils.xyz import load_itk


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    
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
def processSingle3D(imagePath):
    lung3D,origin,spacing=load_itk(imagePath)
    mask_lung3D = segment_lung_mask(lung3D)
    if mask_lung3D.sum() ** (1.0 / 3)>50:
        #save the segmented 3Dlung
        contain = imagePath.split('/')[-2]
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
    contain=img_folder.split('/')[-2]
    out_folder =  img_folder.replace(contain,contain+'_segmentedLungs/')
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    imagePaths = glob.glob('{}*.mhd'.format(img_folder))    
    Parallel(n_jobs=cpu_count())(delayed(processSingle3D)(imagePath) for imagePath in imagePaths)

def test3D():
    start_time = time.time()
    print '{} - start Processing'.format(time.strftime("%H:%M:%s"))
    img_folder = '../lunadata/rawdata/'
    # processfolder(img_folder)
    imagePaths = glob.glob('{}*.mhd'.format(img_folder))
    for i in range(len(imagePaths)):
        imagePath=imagePaths[i]
        lung3D,origin,spacing=load_itk(imagePath)
        mask_lung3D = segment_lung_mask(lung3D, True)
        plot_ct_scan(lung3D, name='origin{}.png'.format(i), plot=False)
        plot_ct_scan(mask_lung3D, name='mask{}.png'.format(i), plot=False)
    print '{} - Processing took {} seconds'.format(time.strftime("%H:%M:%s"),np.floor(time.time()-start_time))

if __name__ == '__main__':
    start_time = time.time()
    print '{} - start Processing'.format(time.strftime("%H:%M:%s"))
    img_folder = '../lunadata/rawdata/'
    processfolder(img_folder)
    print '{} - Processing took {} seconds'.format(time.strftime("%H:%M:%s"),np.floor(time.time()-start_time))
    # test3D()

