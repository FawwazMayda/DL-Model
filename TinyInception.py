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

def inceptionType1(input):
    #Mengingat Konfigurasi Keras bisa channel first atau last
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    # mixed 0: 35 x 35 x 256
    branch1x1 = conv2d_bn(input, 64, 1, 1)

    branch5x5 = conv2d_bn(input, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(input, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(input)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')
    return x

def inception_dim(input):
    #Mengingat Konfigurasi Keras bisa channel first atau last
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1


     branch3x3 = conv2d_bn(input, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(input, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(input)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed3')

    return x


def inceptionV3Base(input):

    x = conv2d_bn(input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)