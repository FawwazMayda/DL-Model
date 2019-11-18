import numpy as np 
import keras
from keras import backend as K 

WEIGHTS_PATH = 'https://github.com/kentsommer/keras-inceptionV4/releases/download/2.1/inception-v4_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/kentsommer/keras-inceptionV4/releases/download/2.1/inception-v4_weights_tf_dim_ordering_tf_kernels_notop.h5'

def preprocess_input(x):
    x = np.divide(x,255.0)
    x = np.subtract(x , 0.5)
    x = np.multiply(x, 2.0)
    return x