import config
import os
import numpy as np

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from segmentation.resnet50 import ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2

def u_net_model(n_classes = 1, input_height=576,
          input_width=576):

    # Get resnet
    if config.segmodel == "RS50":
        img_input, levels = ResNet50(input_shape = (input_height, input_width, 1), classes=2)
    elif config.segmodel == "RS50V":
        img_input, levels = ResNet50V2(input_shape = (input_height, input_width, 1), classes=2)
    elif config.segmodel == "RS101":
        img_input, levels = ResNet101(input_shape = (input_height, input_width, 1), classes=2)
    elif config.segmodel == "RS101V":
        img_input, levels = ResNet101V2(input_shape = (input_height, input_width, 1), classes=2)
    elif config.segmodel == "RS152":
        img_input, levels = ResNet152(input_shape = (input_height, input_width, 1), classes=2)
    else:
        img_input, levels = ResNet152V2(input_shape = (input_height, input_width, 1), classes=2)
    
    [f1, f2, f3, f4, f5] = levels

    o = f4

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(512, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f3], axis=-1))
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f2], axis=-1))
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(128, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f1], axis=-1))
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(64, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(32, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)

    o = Conv2D(n_classes, (3, 3), padding='same')(o)

    o_shape = Model(img_input, o).output_shape
    i_shape = Model(img_input, o).input_shape

    output_height = o_shape[1]
    output_width = o_shape[2]
    input_height = i_shape[1]
    input_width = i_shape[2]
    n_classes = o_shape[3]
    o = (Reshape((output_height * output_width, -1)))(o)

    o = (Activation('sigmoid'))(o)
    model = Model(img_input, o)
    model.output_width = output_width
    model.output_height = output_height
    model.n_classes = n_classes
    model.input_height = input_height
    model.input_width = input_width

    return model
