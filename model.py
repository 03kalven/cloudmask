import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import backend as keras

def get_model(w, d):
    '''
    Returns the cloud_mask model for a given width w and depth d. Works for 668x668x3 inputs.

    Parameters:
    w -- width of the first convolutions in the model
    d -- depth of the first convolutions in the model
    '''
    inputs = Input((668,668,3))
    conv = Lambda(lambda x: x / 255)(inputs)

    skip_layers = []

    for i in range(d):
        conv = Conv2D(w * 2 ** i, 3, activation = 'relu', padding = 'valid', kernel_initializer='he_normal')(conv)
        conv = BatchNormalization()(conv)
        conv = Conv2D(w * 2 ** i, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv)
        conv = BatchNormalization()(conv)
        skip_layers.append(conv)
        conv = MaxPooling2D(pool_size=(2, 2))(conv)

    conv = Conv2D(w * 2 ** d, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv)
    conv = BatchNormalization()(conv)
    conv = Conv2D(w * 2 ** d, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv)
    conv = BatchNormalization()(conv)

    for i, pool in enumerate(reversed(skip_layers)):
        conv = Conv2DTranspose(w * 2 ** (d - 1 - i), (2,2), strides = 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)
        conv = concatenate([Cropping2D((int((pool.shape[1] - conv.shape[1]) / 2),int((pool.shape[2] - conv.shape[2]) / 2)))(pool),conv],axis=3)
        conv = Conv2D(w * 2 ** (d - 1 - i), 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv)
        conv = BatchNormalization()(conv)
        conv = Conv2D(w * 2 ** (d - 1 - i), 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv)
        conv = BatchNormalization()(conv)

    conv = Conv2D(1, 1, activation = 'sigmoid')(conv)
    model = Model(inputs = inputs, outputs = conv)
    return model

def weighted_binary_crossentropy(true, pred, weight_zero = 0.03688):
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