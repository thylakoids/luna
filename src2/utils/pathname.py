import os
datafolder = '../lunadata/'
def segmentedLungs_folder(contain):
    return datafolder+contain+'_segmentedLungs/'
def input_folder(contain):
    return datafolder+contain+'/'
def slices_folder(contain):
    return datafolder+contain+'_slices/'
def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
def mkdir_iter(path):
    if os.path.exists(path):
        return
    path1=os.path.split(path)
    mkdir_iter(path1[0])
    mkdir(path)