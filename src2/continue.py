import h5py
import numpy as np

from keras.models import load_model
from keras.utils import multi_gpu_model
from loadDataTrain import generate_data_from_file
from UNET import dice_coef,dice_coef_loss
from utils.pathname import slices_folder
from config import conf


def main(gpu=2):
    contains = ['subset{}'.format(i) for i in range(8)]
    contains_val = ['subset{}'.format(8)]
    contains_test = ['subset{}'.format(9)]
    train_data=generate_data_from_file([slices_folder(contain) for contain in contains])
    validation_data = generate_data_from_file([slices_folder(contain) for contain in contains_val])
    test_data = generate_data_from_file([slices_folder(contain) for contain in contains_test])
    steps=conf.STEPS
    #load history, get initial_eposh
    f = h5py.File('Unet-history.h5','r+')
    dice = f['dice_coef']
    initial_eposh = dice.shape[0]



    #load model,and trans to gpu_model
    model = load_model("Unet-model.h5", custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    if gpu>=2:
        gpu_model = multi_gpu_model(model,gpus=gpus)
    model.summary()
    gpu_model.summary()
    adam = Adam(lr=1.0e-6)
    gpu_model.compile(optimizer=adam, loss=dice_coef_loss, metrics=[dice_coef])

    #evaluate
    loss, accuracy = gpu_model.evaluate_generator(test_data, steps=steps)
    print('\ntest loss', loss)
    print('dice_coef', accuracy)



    #continue train
    history=gpu_model.fit_generator(train_data,steps_per_epoch=steps*8,epochs=conf.EPOCHS+initial_eposh,verbose=2,
                                validation_data=validation_data,validation_steps=steps,initial_epoch=initial_eposh)
    # save history
    dice = np.append(f['dice_coef'],history.history['dice_coef'])
    val_dice = np.append(f['val_dice_coef'],history.history['val_dice_coef'])
    del f['dice_coef'], f['val_dice_coef']

    f['dice_coef'] = dice
    f['val_dice_coef'] = val_dice
    f.close

    test_data = generate_data_from_file([slices_folder(contain) for contain in contains_test])
    loss,accuracy = gpu_model.evaluate_generator(test_data,steps=steps)
    print('\ntest loss',loss)
    print('dice_coef',accuracy)
    #save then delete model
    model.save('Unet-model4.h5')
    del model
    del gpu_model
if __name__=='__main__':
    main()
