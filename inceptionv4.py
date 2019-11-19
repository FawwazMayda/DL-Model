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
    x = keras.layers.Convolution2D(nb_filter, (num_row, num_col),
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      kernel_regularizer=keras.regularizers.l2(0.00004),
                      kernel_initializer=keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None))(x)
    x = keras.layers.BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(x)
    x = keras.layers.Activation('relu')(x)

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

    b3 = keras.layers.AveragePooling2D((3,3), strides=(1,1), padding='same')(input)
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


def inceptionV4Base(input):
    #Mengingat Konfigurasi Keras bisa channel first atau last
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    net = conv2d_bn(input,32,3,3,strides=(2,2),padding='valid')
    net = conv2d_bn(net,32,3,3,padding='valid')
    net = conv2d_bn(net,64,3,3)



    b0 = keras.layers.MaxPooling2D((3,3),strides=(2,2),padding='valid')(net)
    b1 =  conv2d_bn(net,96,3,3,strides=(2,2),padding='valid')
    net = keras.layers.concatenate([b0,b1],axis=channel_axis)



    b0 = conv2d_bn(net,64,1,1)
    b0 = conv2d_bn(b0,96,3,3,padding='valid')

    b1 = conv2d_bn(net,64,1,1)
    b1 = conv2d_bn(b1,64,1,7)
    b1 = conv2d_bn(b1,64,7,1)
    b1 = conv2d_bn(b1,96,3,3,padding='valid')

    net = keras.layers.concatenate([b0,b1],axis=channel_axis)


    b0 = conv2d_bn(net,192,3,3,padding='valid',strides=(2,2))
    b1 = keras.layers.MaxPooling2D((3,3),strides=(2,2),padding='valid')(net)

    net = keras.layers.concatenate([b0,b1],axis=channel_axis)

    for i in range(4):
        net = inception_a(net)
    
    net = reduction_a(net)

    for i in range(7):
        net = inception_b(net)

    net = reduction_b(net)

    for i in range(5):
        net = inception_c(net)
    
    return net

def inceptionV4(jumlah_kelas,dropout_keep_rate,weights,include_top):
    #Mengingat Konfigurasi Keras bisa channel first atau last
    a,b = input_shape
    if K.image_data_format() == 'channels_first':
        inputs = keras.layers.Input((3,229,229))
    else:
        inputs = keras.layers.Input((299,299,3))

    x =  inceptionV4Base(inputs)

    if include_top:
        x = keras.layers.AveragePooling2D((8,8),padding='valid')(x)
        x = keras.layers.Dropout(dropout_keep_rate)(x)
        #Hasil Embeddings
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(jumlah_kelas,activation='softmax')(x)

    model = keras.models.Model(inputs,x,name='InceptionV4')

    # load weights
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_first':
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
        if include_top:
            weights_path = get_file(
                'inception-v4_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='9fe79d77f793fe874470d84ca6ba4a3b')
        else:
            weights_path = get_file(
                'inception-v4_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='9296b46b5971573064d12e4669110969')
        model.load_weights(weights_path, by_name=True)

    return model

    














    






