
def zero_center(image,PIXEL_MEAN=0.17):
    '''
    zero center
    :param image: input image
    :param PIXEL_MEAN: 0 for not doing zero center
    :return: zero-centered image
    '''
    image = image - PIXEL_MEAN
    return image


def normalizePlanes(npzarray):
    maxHU = 400.
    minHU = -1000.

    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1.
    npzarray[npzarray < 0] = 0.
    return npzarray

def normalize(lung,lung_mask):
    lung = normalizePlanes(lung)

    lung_mean = lung[lung_mask==1].mean()
    lung_std = lung[lung_mask==1].std()

    lung[lung_mask==0] = lung_mean-1.2*lung_std
    lung = lung - lung_mean
    lung = lung/lung_std
    return lung
