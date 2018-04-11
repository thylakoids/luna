
def zero_center(image,PIXEL_MEAN=0.11):
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