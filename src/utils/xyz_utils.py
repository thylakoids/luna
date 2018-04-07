import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
def plot_ct_scan(scan):
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(25, 25))
    for i in range(0, scan.shape[0], 5):
        plots[int(i / 20), int((i % 20) / 5)].axis('off')
        plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i], cmap=plt.cm.bone) 
    for j in range(4):
        plots[int(i / 20), j].axis('off')
    plt.show()
def plot_3d(image, threshold=-300): #########
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(1,2,0)
    
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


if __name__ == "__main__":
    image, origin, spacing = load_itk('../../lunadata/rawdata/1.3.6.1.4.1.14519.5.2.1.6279.6001.317087518531899043292346860596.mhd')
    print 'slice0:\n',image[:,:,0]
    print 'Shape:', image.shape
    print 'Origin:', origin
    print 'Spacing:', spacing

    # plt.hist(image.flatten(), bins=80, color='c')
    # plt.xlabel("Hounsfield Units (HU)")
    # plt.ylabel("Frequency")
    # plt.show()

    # # Show some slice in the middle
    # plt.imshow(image[240], cmap=plt.cm.gray)
    # plt.colorbar()
    # plt.show()

    # plot_ct_scan(image)
    plot_3d(image)

