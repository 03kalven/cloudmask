import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import backend as keras

def getModel(w):
    '''
    Returns the cloudMask model for a given width w. Works for 668x668x3 inputs. This input size was chosen because it minimized the overlap needed during the subdividing process as compared to similar input sizes.

    Parameters:
    w -- width of the first convolutions in the model
    '''
    inputs = Input((668,668,3))
    s = Lambda(lambda x: x / 255)(inputs)
    conv1 = Conv2D(w, 3, activation = 'relu', padding = 'valid', kernel_initializer='he_normal')(s)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(w, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(2*w, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(2*w, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(4*w, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(4*w, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(8*w, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(8*w, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(16*w, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(16*w, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = Conv2DTranspose(8*w, (2,2), strides = 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    merge6 = concatenate([experimental.preprocessing.CenterCrop(up6.get_shape()[1],up6.get_shape()[2])(conv4),up6], axis = 3)
    conv6 = Conv2D(8*w, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(8*w, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2DTranspose(4*w, (2,2), strides = 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    merge7 = concatenate([experimental.preprocessing.CenterCrop(up7.get_shape()[1],up7.get_shape()[2])(conv3),up7], axis = 3)
    conv7 = Conv2D(4*w, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(4*w, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2DTranspose(2*w, (2,2), strides = 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    merge8 = concatenate([experimental.preprocessing.CenterCrop(up8.get_shape()[1],up8.get_shape()[2])(conv2),up8], axis = 3)
    conv8 = Conv2D(2*w, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(2*w, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2DTranspose(w, (2,2), strides = 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    merge9 = concatenate([experimental.preprocessing.CenterCrop(up9.get_shape()[1],up9.get_shape()[2])(conv1),up9], axis = 3)
    conv9 = Conv2D(w, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(w, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)
    return model

def staticWeightedBincrossentropy(true, pred, weight_zero = 0.03693):
    '''
    A custom loss function that performs binary cross entropy but weights certain predictions using weight_zero. The default weight_zero was derived from Y_Train.sum()/Y_Train.size

    Parameters:
    true -- the true value tensor
    pred -- the prediction value tensor
    weight_zero -- a scalar to weight predictions on zero pixels. Helpful to keep < 1 in unbalanced datasets (like this one) where a zero prediction is much more common than a one (default 0.03688)
    '''
    true = tf.cast(true, tf.float32)
    pred = tf.cast(pred, tf.float32)
    bin_crossentropy = keras.binary_crossentropy(true, pred)
    
    # apply the weights
    weights = true * (1. - weight_zero) + (1. - true) * weight_zero
    weighted_bin_crossentropy = weights * bin_crossentropy

    return keras.mean(weighted_bin_crossentropy)