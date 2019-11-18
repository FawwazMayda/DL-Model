import numpy as np 
import keras

def conv2d_bn(x, nb_filter, num_row, num_col,
              padding='same', strides=(1, 1), use_bias=False):
    #Mengingat Konfigurasi Keras bisa channel first atau last
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    x = keras.layers.Convolution2D(nb_filter, (num_row, num_col),
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      kernel_regularizer=keras.regularizers.l2(0.00004),
                      kernel_initializer=keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None))(x)
    x = keras.layers.BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(x)
    x = keras.layers.Activation('relu')(x)

    return x