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

def conv2d_bn(x, nb_filter, num_row, num_col,
              padding='same', strides=(1, 1), use_bias=False):
    #Mengingat Konfigurasi Keras bisa channel first atau last
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    x = Convolution2D(nb_filter, (num_row, num_col),
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      kernel_regularizer=regularizers.l2(0.00004),
                      kernel_initializer=initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None))(x)
    x = BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(x)
    x = Activation('relu')(x)
    return x

def inception_a(input):
    #Mengingat Konfigurasi Keras bisa channel first atau last
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    #1x1
    b0 = conv2d_bn(input, 96,1,1)
    #1x1 -> 3x3
    b1 = conv2d_bn(input,64,1,1)
    b1 = conv2d_bn(b1,96,3,3)
    #1x1 -> 3x3 -> 3x3
    b2 = conv2d_bn(input,64,1,1)
    b2 = conv2d_bn(b2,96,3,3)
    b2 = conv2d_bn(b2,96,3,3)

    b3 = keras.layers.AveragePooling2D((3,3),strides=(1,1),
    padding='same')(input)
    b3 = conv2d_bn(b3,96,1,1)

    x= keras.layers.concatenate([b0,b1,b2,b3],axis=channel_axis)
    return x
def reduction_a(input):
    #Mengingat Konfigurasi Keras bisa channel first atau last
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    b0 = conv2d_bn(input,384,3,3,strides=(2,2),padding='valid')

    b1 = conv2d_bn(input,192,1,1)
    b1 = conv2d_bn(b1,224,3,3)
    b1 = conv2d_bn(b1,256,3,3,strides=(2,2),padding='valid')

    b2 = keras.layers.MaxPooling2D((3,3),strides=(2,2),padding='valid')(input)

    x = keras.layers.concatenate([b0,b1,b2],axis=channel_axis)
    return x

def inception_b(input):
    #Mengingat Konfigurasi Keras bisa channel first atau last
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    b0 = conv2d_bn(input,384,1,1)

    b1 = conv2d_bn(input,192,1,1)
    b1 = conv2d_bn(b1,224,7,1)
    b1 = conv2d_bn(b1,256,1,7)

    b2 = conv2d_bn(input,192,1,1)
    b2 = conv2d_bn(b2,192,7,1)
    b2 = conv2d_bn(b2, 224, 1, 7)
    b2 = conv2d_bn(b2, 224, 7, 1)
    b2 = conv2d_bn(b2, 256, 1, 7)

    b3 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input)
    b3 = conv2d_bn(b3,128,1,1)

    x = keras.layers.concatenate([b0,b1,b2,b3],axis=channel_axis)
    return x

def reduction_b(input):
    #Mengingat Konfigurasi Keras bisa channel first atau last
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    b0 = conv2d_bn(input,192,1,1)
    b0 = conv2d_bn(b0,192,3,3,strides=(2,2),padding='valid')

    b1 = conv2d_bn(input,256,1,1)
    b1 = conv2d_bn(b1,256,1,7)
    b1 = conv2d_bn(b1,320,7,1)
    b1 = conv2d_bn(b1,320,3,3,strides=(2,2),padding='valid')

    b2 = keras.layers.MaxPooling2D((3,3),strides=(2,2),padding='valid')(input)

    x = keras.layers.concatenate([b0,b1,b2],axis=channel_axis)
    return x

def inception_c(input):
    #Mengingat Konfigurasi Keras bisa channel first atau last
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    b0 = conv2d_bn(input,256,1,1)

    b1 = conv2d_bn(input,384,1,1)
    b10 = conv2d_bn(b1,256,1,3)
    b11 = conv2d_bn(b1,256,3,1)
    b1 = keras.layers.concatenate([b10,b11],axis=channel_axis)

    b2 = conv2d_bn(input,384,1,1)
    b2 = conv2d_bn(b2,448,3,1)
    b2 = conv2d_bn(b2,512,1,3)
    b20 = conv2d_bn(b2,256,1,3)
    b21 = conv2d_bn(b2,256,3,1)
    b2 = keras.layers.concatenate([b20,b21],axis=channel_axis)

    b3 = keras.layers.AveragePooling2D((3,3),strides=(1,1),padding='same')(input)
    b3 = conv2d_bn(b3,256,1,1)
    
    x = keras.layers.concatenate([b0,b1,b2,b3],axis=channel_axis)


    







    






