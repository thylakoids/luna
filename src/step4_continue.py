import h5py

from keras.models import load_model
from step3_train import load_data
from step2_UNET import dice_coef,dice_coef_loss
from keras import backend as  K

x_train,y_train,x_test,y_test=load_data()


model = load_model("Unet.h5", custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})

lr=K.get_value(model.optimizer.lr)
K.set_value(model.optimizer.lr, lr/10)
print K.get_value(model.optimizer.lr)
history=model.fit(x_train,y_train,batch_size=4,epochs=20,verbose=2,validation_split=0.1) #0.1 for validate
# to do : save histoty,plot history
# f=h5py.File("model_history.hdf5","w")
# f['history']=history
# f.close



loss,accuracy = model.evaluate(x_test,y_test)
print('\ntest loss',loss)
print('dice_coef',accuracy)
#save then delete model
model.save('Unet-2.h5')
del model



