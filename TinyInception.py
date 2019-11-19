import numpy as np 
import keras
from keras import backend as K

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

    branch_pool = keras.layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(input)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = keras.layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis)
    return x

def inceptionType2(input):
    #Mengingat Konfigurasi Keras bisa channel first atau last
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    branch1x1 = conv2d_bn(input, 192, 1, 1)

    branch7x7 = conv2d_bn(input, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(input, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = keras.layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(input)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = keras.layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis)
    return x

def inceptionType3(input):
    #Mengingat Konfigurasi Keras bisa channel first atau last
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1


        branch1x1 = conv2d_bn(input, 320, 1, 1)

        branch3x3 = conv2d_bn(input, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = keras.layers.concatenate(
            [branch3x3_1, branch3x3_2],
            axis=channel_axis)

        branch3x3dbl = conv2d_bn(input, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = keras.layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = keras.layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(input)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = keras.layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis)
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

    branch_pool = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(input)
    x = keras.layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis)

    return x
    
def inception_dim2(input):
    #Mengingat Konfigurasi Keras bisa channel first atau last
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    branch3x3 = conv2d_bn(input, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(input, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(input)
    x = keras.layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=channel_axis)
    return x


def inceptionV3Base(jumlah_kelas):


    inputs = keras.layers.Input((64,64,3))
    net = conv2d_bn(inputs, 32, 3, 3, strides=(2, 2), padding='valid')
    net = conv2d_bn(net, 32, 3, 3, padding='valid')
    net = conv2d_bn(net, 64, 3, 3)

    for i in range(2):
        net = inceptionType1(net)
    net = inception_dim(net)

    for i in range(3):
        net = inceptionType2(net)
    net = inception_dim2(net)
    net = inceptionType3(net)

    net = keras.layers.AveragePooling2D()(net)
    if jumlah_kelas>2:
        net = keras.layers.Dense(2,activation='sigmoid')(net)
    else:
        net = keras.layers.Dense(jumlah_kelas,activation='softmax')(net)
    model = keras.models.Model(inputs,net,name='TinyInception')
    return model
