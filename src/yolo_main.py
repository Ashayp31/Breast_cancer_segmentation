"""
Retrain the YOLO model for your own dataset.
Code taken from yolo3 implementation code at : https://github.com/qqwweee/keras-yolo3
Code adapted to work with DDSM dataset changing training input process and preprocessing
"""

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from keras_yolo3.yolo3.model import yolo_body, tiny_yolo_body, yolo_loss
from data_operations.dataset_feed import bbox_dataset

import config


def _main():
    annotation_path = '../data/CBIS-DDSM-mask/bbox_groud_truth.txt'
    anchors_path = 'keras_yolo3/model_data/yolo_anchors.txt'
    num_classes = 1
    anchors = get_anchors(anchors_path)

    input_shape = (config.VGG_IMG_SIZE["HEIGHT"], config.VGG_IMG_SIZE["WIDTH"]) # multiple of 32, hw

    model = create_model(input_shape, anchors, num_classes,
            freeze_body=1, weights_path='keras_yolo3/model_data/darknet53_weights.h5') # make sure you know what you freeze

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.25
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    train_dataset = bbox_dataset(lines[:num_train])
    val_dataset = bbox_dataset(lines[num_train:])
    
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        model.fit(x=train_dataset,
                  validation_data=val_dataset,
                  epochs=50)


    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        model.fit(x=train_dataset,
                  validation_data=val_dataset,
                  epochs=60,
                  callbacks=[reduce_lr, early_stopping])
        model.save_weights('../saved_models/yolo_v3_trained_weights_final.h5')

    # Further training if needed.


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': 1, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)
    print(model)
    print(*model_body.output)
    print(*y_true)
    print("HERE")
    return model
                                                           
                                                                             
                                                                                                                                
if __name__ == '__main__':
    _main()
