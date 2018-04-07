import os
datafolder = '../lunadata/'
def segmentedLungs_folder(contain):
    return datafolder+contain+'_segmentedLungs/'
def input_folder(contain):
    return datafolder+contain+'/'
def slicesLungs_folder(contain):
    return datafolder+contain+'_1_1_1mm_slices_lung/'
def slicesNodule_folder(contain):
    return datafolder+contain+'_1_1_1mm_slices_nodule/'
def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)