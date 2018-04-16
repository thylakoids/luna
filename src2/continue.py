import h5py

from keras.models import load_model
from loadDataTrain import generate_data_from_file
from UNET import dice_coef,dice_coef_loss
from utils.pathname import slices_folder

def main():
    contains = ['subset{}'.format(i) for i in range(8)]
    contains_val = ['subset{}'.format(8)]
    contains_test = ['subset{}'.format(9)]
    train_data=generate_data_from_file([slices_folder(contain) for contain in contains])
    validation_data = generate_data_from_file([slices_folder(contain) for contain in contains_val])
    test_data = generate_data_from_file([slices_folder(contain) for contain in contains_test])
    steps=1000
    #load model
    model = load_model("Unet-model_20.h5", custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    loss, accuracy = model.evaluate_generator(test_data, steps=100)
    print('\ntest loss', loss)
    print('dice_coef', accuracy)

    #continue train
    history=model.fit_generator(train_data,steps_per_epoch=steps*8,epochs=25,verbose=2,
                                validation_data=validation_data,validation_steps=steps,initial_epoch=20)
    #save history
    f=h5py.File("Unet-history_25.h5","w")
    f['dice_coef']=history.history['dice_coef']
    f['val_dice_coef']=history.history['val_dice_coef']
    f.close

    loss,accuracy = model.evaluate_generator(test_data,steps=steps)
    print('\ntest loss',loss)
    print('dice_coef',accuracy)
    #save then delete model
    model.save('Unet-model_25.h5')
    del model
if __name__=='__main__':
    main()
