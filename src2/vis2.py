from loadDataTrain import *
from numpy import argsort
from utils.xyz import load_predict, world_2_voxel
from utils.normalize import normalize
from utils.visualization import array2video,impose2
from skimage.feature import blob_dog


import re
import pickle
import gzip

import pandas as pd
import numpy as np
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
    def predict(self,model):
        '''
        note: load model on every 2d pic, may slow down the process!!!!
        '''
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
    model = load_model("Unet-model.h5", custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
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
            for slicePath in slicePaths:
                print os.path.basename(slicePath)
                lung, lung_mask, nodule_mask, origin, space =load_slice(slicePath)
                AI = lung2DAI(lung,lung_mask,nodule_mask)
                noduel_mask_pre = AI.predict(model)

                lung3D.append(lung)
                lung_mask3D.append(lung_mask)
                nodule_mask3D.append(nodule_mask)
                nodule_mask_pre3D.append(noduel_mask_pre)
            print np.array(lung3D).shape
            #save it
            savePath = os.path.join(predict_folder(contain),imagename+'.pkl.gz')
            file = gzip.open(savePath,'wb')
            pickle.dump(np.array(lung3D),file,protocol=-1)
            pickle.dump(np.array(lung_mask3D),file,protocol=-1)
            pickle.dump(np.array(nodule_mask3D),file,protocol=-1)
            pickle.dump(np.array(nodule_mask_pre3D),file,protocol=-1)
            pickle.dump(origin,file,protocol=-1)
            pickle.dump(space,file,protocol=-1)
            file.close()
def createvideo():
    '''
    creat video for demonstration
    '''
    contains = conf.FOLDERS
    for contain in contains:
        paths = glob.glob(os.path.join(
            predict_folder(contain),
            '*.pkl.gz'))
        for path in paths:
            lung3D,lung_mask3D,nodule_mask3D,nodule_mask_pre3D=load_predict(path)
            lung3D_norm = normalizePlanes(lung3D)
            #1
            video1 = np.stack([lung3D_norm]*3,axis = -1)

            #2
            # lungn3D=(lungn3D+1)/2
            lung3D_norm[lung_mask3D==0]=0
            video2 = impose2(lung_mask3D,nodule_mask3D)
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

            #save 1
            savePath = path.replace('.pkl.gz','_3.mp4')
            video = video[20:-20,:,:,:]
            array2video(video,savePath)
            # #save 2
            # newvideo=cumulative(video)
            # savePath = path.replace('.pkl.gz','2.mp4')
            # array2video(video,savePath)



def _evaluaton_annotation(gts,pres):
    gts_evaluation = np.zeros((gts.shape[0],1))# 0:FN, 1:TP
    pres_evaluation = np.zeros((pres.shape[0],1))# 0:FP, 1:TP, 2:irrelevant
    for i in range(gts.shape[0]):
        for j in range(pres.shape[0]):
            if np.linalg.norm(gts[i,:3]-pres[j,:3])<gts[i,3]:
                if gts_evaluation[i]==0:
                    gts_evaluation[i]=1
                    pres_evaluation[j]=1
                # this gt already hit by other prediction,mark this one irrelevent
                else:
                    pres_evaluation[j]=2
    # pre_P = pres_evaluation.sum()
    # pre_N = (1-pres_evaluation).sum()
    # gt_P = gts_evaluation.sum()
    # gt_N =(1-gts_evaluation).sum()
    return gts_evaluation,pres_evaluation

def evaluation():
    contains = conf.FOLDERS
    annotations_path = conf.ANNOTATION_PATH
    annotations = pd.read_csv(annotations_path)
    gt_P=0
    gt_N=0
    pre_N=0
    pre_P=0
    for contain in contains:
        paths = glob.glob(os.path.join(predict_folder(contain),'*.pkl.gz'))
        for path in paths:
            lung3D,lung_mask3D,nodule_mask3D,nodule_mask_pre3D,origin,spacing=load_predict(path)
            imageName = os.path.basename(path).replace('.pkl.gz', '')
            image_annotations = annotations[annotations['seriesuid'] == imageName]
            image_annotations_voxel = []
            for ca in image_annotations.values:
                annotation_voxel = world_2_voxel(np.array((ca[3], ca[2], ca[1])),origin,spacing)
                #coordinate and radius
                image_annotations_voxel.append(np.hstack((annotation_voxel,ca[4]/2)))
            image_annotations_voxel = np.array(image_annotations_voxel)
            image_predict_voxel = blob_dog(nodule_mask_pre3D,threshold=0.1,min_sigma=1.5,max_sigma=15)
            gts,pres=_evaluaton_annotation(image_annotations_voxel,image_predict_voxel)
            gt_P+=(gts==1).sum()
            gt_N+=(gts==0).sum()
            pre_P+=(pre==1).sum()
            pre_N+=(pre==0).sum()

    recall = 1.0*gt_P/(gt_P+gt_N)
    precision = 1.0*pre_P/(pre_P+pre_N)
    return recall, precision


def cumulative(img,window=30):
    '''
    transparent
    '''
    nz = img.shape[0]
    newimg = []
    for i in range(nz-window):
        slice = img[i:i+window].sum(axis =0)
        slice[slice>1]=1
        newimg.append(slice)
    return np.array(newimg)


if __name__ == '__main__':
    # lung3DAI()
    # createvideo()
    evaluation()