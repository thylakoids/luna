from keras.utils import plot_model

import sys
sys.path.append("..")
from step2_UNET import unet_model

model = unet_model()
plot_model(model,to_file='unet_model.png',show_shapes=True,show_layer_names=False)