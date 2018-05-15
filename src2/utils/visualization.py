from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import pyplot as plt
from skimage import measure
from skimage.color import gray2rgb
import scipy.ndimage

import numpy as np
from  scipy import stats

def plot_ct_scan(scan,name=False,plot=True):
    skip=60
    nClom=3
    while scan.shape[0]<=skip*nClom:
        skip=int(skip/2)
    N=skip*nClom

    f,plots = plt.subplots(int(scan.shape[0]/N)+1,nClom,figsize=(25, 25))
    for i in range(0, scan.shape[0], skip):
        plots[int(i / N), int((i % N) / skip)].axis('off')
        plots[int(i / N), int((i % N) / skip)].imshow(scan[i],cmap=plt.cm.gray) 
    for j in range(nClom):
        plots[int(i / N), j].axis('off')
    if name:
        plt.savefig(name)
    if plot:
        plt.show()
def plot_3d(image, threshold=-300,zoom=0.5): #########
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    p = scipy.ndimage.interpolation.zoom(p,zoom)
    verts, faces,_,_= measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()
def impose(img,mask,col = np.array([255, 0, 0])/ 255.0,alpha=0.5):
    '''
    impose 2 gray img into one rbg img
    '''
    img_lung = img.copy()
    img_nodule = img.copy()
    img_lung[mask==1]=0
    img_nodule[mask==0]=0


    img_nodule_3c = np.stack(
        [img_nodule]*3,axis=-1)
    img_lung_3c = np.stack(
        [img_lung]*3,axis=-1)
    mask_3c = np.stack(
        (mask * col[0], mask * col[1],mask * col[2]),axis=-1)
    img_impose = img_lung_3c+(1-alpha)*img_nodule_3c+alpha*mask_3c
    return img_impose
def impose2(img,mask,col = np.array([255, 0, 0])/ 255.0,alpha=0.5):
    '''
    impose 2 gray img into one rbg img
    '''
    img_lung = img.copy()
    img_nodule = img.copy()
    img_lung[mask==1]=0
    img_nodule[mask==0]=0


    img_lung_3c = np.stack([img_lung]*3,axis = -1)
    img_nodule_3c=np.stack(
        (img_nodule*col[0],img_nodule*col[1],img_nodule*col[2]),axis = -1)

    mask_3c = np.stack([mask]*3,axis = -1)

    blended = mask_3c*img_nodule_3c+(1-mask_3c)*img_lung_3c
    return blended


def myball(radius,size=4, dtype=np.float32):
    draw_radius=radius*size
    n = 2 * draw_radius + 1
    Z, Y, X = np.mgrid[-draw_radius:draw_radius:n * 1j,
                       -draw_radius:draw_radius:n * 1j,
                       -draw_radius:draw_radius:n * 1j]
    s = X ** 2 + Y ** 2 + Z ** 2
    if size!=1:
        mask = stats.norm.pdf(np.sqrt(s)/radius)
        return np.array(mask/mask.max(), dtype=dtype)
    else:
        return np.array(s <= radius * radius, dtype=dtype)


if __name__=='__main__':
    mask = myball(5)
    plt.imshow(mask[20])
    plt.show()



