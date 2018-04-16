from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import pyplot as plt
from skimage import measure

import scipy.ndimage

import numpy as np

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
def impose(img,mask):
    image_nodule = img.copy()
    image_lung = img.copy()
    image_nodule[mask == 0] = 0
    image_lung[mask != 0] = 0

    image_3c = np.stack(
        (image_lung, image_lung, image_lung)).transpose(1, 2, 0)
    image_nodule_3c = np.stack(
        (image_nodule, image_nodule, image_nodule)).transpose(1, 2, 0)
    alpha = 0.6
    col = np.array([222, 129, 0]) / 255.0
    # col = np.array([175, 99, 37]) / 255.0 / alpha
    nodule_3c = np.stack((mask * col[0], mask * col[1],
                          mask * col[2])).transpose(1, 2, 0)
    return image_3c + (1 - alpha) * image_nodule_3c + alpha * nodule_3c
