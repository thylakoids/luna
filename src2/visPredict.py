from keras.models import load_model
from loadDataTrain import generate_data_from_file
from UNET import dice_coef,dice_coef_loss
from utils.pathname import slices_folder
import matplotlib.pyplot as plt


def vispre():
    #test data
    contains_test = ['subset{}'.format(9)]
    test_data = generate_data_from_file([slices_folder(contain) for contain in contains_test])
    #load model
    model = load_model("Unet-model.h5", custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})


    X,Y = test_data.next()
    Y_pre = model.predict(X,verbose=0)
    _,plots = plt.subplots(3,4)
    for i in range(X.shape[0]):
        plots[0,i].imshow(X[i][0])
        plots[1,i].imshow(Y[i][0])
        plots[2,i].imshow(Y_pre[i][0])
    plt.savefig('predict.png')

if __name__ =='__main__':
    vispre()

