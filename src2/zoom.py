#%%
from scipy import ndimage as nd
from skimage.transform import resize,rescale
from  matplotlib import pyplot as plt

#%%
p=plt.imread('slice.png')
p=p[:,:,0]
plt.imshow(p,cmap='gray')
plt.show()

#%%
p1=nd.interpolation.zoom(p,2)
plt.imshow(p1,cmap='gray')
plt.show()

#%%
p1=nd.interpolation.zoom(p,0.5,order=0)
plt.imshow(p1,cmap='gray')
plt.show()

#%%
p1=nd.zoom(p,2)
plt.imshow(p1,cmap='gray')
plt.show()

#%%
p1=nd.zoom(p,0.5,order=0)
plt.imshow(p1,cmap='gray')
plt.show()

#%%
p1=rescale(p,0.5,order=0)
plt.imshow(p1,cmap='gray')
plt.show()

#%%
p1=rescale(p,2)
plt.imshow(p1,cmap='gray')
plt.show()