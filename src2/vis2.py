from loadDataTrain import *
from numpy import argsort
from utils.xyz import load_predict
from utils.normalize import normalize
from utils.visualization import array2video,impose2

import re
import pickle
import gzip
class lung2DAI():
    '''
    padding and cropping image

    (512,512)--cropping-->(h,w)--resize-->(512,512)
    (512,512)<--padding--(h,w)<--resize--(512,512)
    '''
    def __init__(self,lung,lung_mask,nodule_mask):
        self.size = 512.0
        self.lung= lung
        self.lung_mask=lung_mask
        self.nodule_mask=nodule_mask
        pos = np.where(lung_mask >0)
        self.x = pos[0].min()
        self.y = pos[1].min()
        self.h = pos[0].max()-pos[0].min()
        self.w = pos[1].max()-pos[1].min()
        self.resize = np.array([1.0*self.size/self.h,1.0*self.size/self.w])
        self.nlung = self._normalize()
    def _cropResize(self):
        lung = self.nlung[self.x:self.x+self.h,self.y:self.y+self.w]
        nodule_mask = self.nodule_mask[self.x:self.x+self.h,self.y:self.y+self.w]
        lung = nd.interpolation.zoom(lung, self.resize,order=0)
        nodule_mask = nd.interpolation.zoom(nodule_mask, self.resize,order=0)
        return lung, nodule_mask
    def traindata(self):
        lung,nodule_mask = self._cropResize()
        return lung[np.newaxis,np.newaxis,:],nodule_mask[np.newaxis,np.newaxis,:]
    def predict(self):
        model = load_model("Unet-model.h5", custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
        Xr,Yr = self.traindata()
        Yr_pre = model.predict(Xr,verbose=0)
        Yo_pre = self._resizePadding(Yr_pre.squeeze())
        return Yo_pre
    def showpredict(self):
        plt.subplot(1,3,1)
        plt.imshow(self.lung,cmap='gray')
        plt.subplot(1,3,2)
        plt.imshow(self.nodule_mask,cmap='gray')
        plt.title('Ground truth')
        plt.subplot(1,3,3)
        plt.imshow(self.predict(),cmap='gray')
        plt.title('Predict')
        plt.show()
        

    def _resizePadding(self,nodule_mask):
        nodule_mask = nd.interpolation.zoom(nodule_mask, 1.0/self.resize,order=0)
        zero_mask = np.zeros(self.nodule_mask.shape)
        zero_mask[self.x:self.x+self.h,self.y:self.y+self.w]=nodule_mask
        return zero_mask
    def _normalize(self):
        lung = normalizePlanes(self.lung)
        lung_mean = lung[self.lung_mask==1].mean()
        lung_std = lung[self.lung_mask==1].std()
        lung[self.lung_mask==0] = lung_mean-1.2*lung_std
        lung = lung - lung_mean
        lung = lung/lung_std
        return lung

def lung3DAI():
    contains = conf.FOLDERS
    for contain in contains:
        mkdir(predict_folder(contain))
        imagenames = glob.glob(os.path.join(
            input_folder(contain),
            '*.mhd'))
        for imagename in imagenames:
            imagename = os.path.basename(imagename).replace('.mhd','')
            slicePaths =glob.glob(os.path.join(
                slices_folder(contain),
                '{}_slice*.pkl.gz'.format(imagename)))

            #sort the slices
            slicebasePaths = [os.path.basename(x) for x in slicePaths]
            index = argsort([int(re.findall('^[\.0-9]+_slice([0-9]+).+gz$',x)[0]) for x in slicebasePaths])
            slicePaths=[slicePaths[x] for x in index]
            lung3D=[]
            lung_mask3D=[]
            nodule_mask3D=[]
            nodule_mask_pre3D=[]
            i = 0
            for slicePath in slicePaths:
                print os.path.basename(slicePath)
                lung, lung_mask, nodule_mask, _,_ =load_slice(slicePath)
                AI = lung2DAI(lung,lung_mask,nodule_mask)
                noduel_mask_pre = AI.predict()

                lung3D.append(lung)
                lung_mask3D.append(lung_mask)
                nodule_mask3D.append(nodule_mask)
                nodule_mask_pre3D.append(noduel_mask_pre)
                i+=1
                if i>=30:
                    pass
            print np.array(lung3D).shape
            #save it
            savePath = os.path.join(predict_folder(contain),imagename+'.pkl.gz')
            file = gzip.open(savePath,'wb')
            pickle.dump(np.array(lung3D),file,protocol=-1)
            pickle.dump(np.array(lung_mask3D),file,protocol=-1)
            pickle.dump(np.array(nodule_mask3D),file,protocol=-1)
            pickle.dump(np.array(nodule_mask_pre3D),file,protocol=-1)
            file.close()
def createvideo():
    contains = conf.FOLDERS
    for contain in contains:
        paths = glob.glob(os.path.join(
            predict_folder(contain),
            '*.pkl.gz'))
        for path in paths:
            lung3D,lung_mask3D,nodule_mask3D,nodule_mask_pre3D=load_predict(path)
            #1
            video1 = np.stack([lung3D]*3,axis = -1)
            video1 = (video1-video1.min())/(video1.max()-video1.min())
            #2
            lungn3D = normalizePlanes(lung3D)
            lungn3D=(lungn3D+1)/2
            lungn3D[lung_mask3D==0]=0
            video2 = impose2(lungn3D,nodule_mask3D)
            #3
            video3 = np.stack([nodule_mask_pre3D]*3,axis = -1)

            # put it togather
            # video = np.ones(())
            videoshape = np.array(video1.shape)
            videoshape[2]=(videoshape[2]*3)+10*2

            video = np.ones(videoshape)
            video[:,:,0:0+512,:]=video1
            video[:,:,512+10:10+512+512,:]=video2
            video[:,:,512+10+512+10:10+512+512+512+10,:]=video3

            savePath = path.replace('.pkl.gz','.mp4')
            array2video(video,savePath)

if __name__ == '__main__':
    createvideo()