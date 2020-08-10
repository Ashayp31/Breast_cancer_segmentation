from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError, binary_crossentropy
from tensorflow.keras.metrics import BinaryAccuracy, MeanIoU
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import config

from tensorflow.keras import backend as K
from model.u_net_RES import u_net_res_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from data_visualisation.output import evaluate, evaluate_segmentation, visualise_examples
from data_operations.dataset_feed import create_dataset_masks, create_dataset_cropped



def train_segmentation_network_incremental(images, image_masks):
    """
    Function to train network in incremental steps increasing image size:
        * Train network with initial ResNET backbone layers frozen
        * Unfreeze all layers and retrain with smaller learning rate
    """
    sizes = [[448,320], [768,544], [1280,864]]
#     sizes = [[224,160], [544,384], [1280,864]]

#     batch_sizes = [12, 4, 1]
    batch_sizes = [8, 4, 1]

    config.VGG_IMG_SIZE["HEIGHT"] = sizes[0][0]
    config.VGG_IMG_SIZE["WIDTH"] = sizes[0][1]
    
    model = u_net_res_model(input_height = config.VGG_IMG_SIZE["HEIGHT"], input_width = config.VGG_IMG_SIZE["WIDTH"])
    
    X_train, X_val, y_train, y_val = train_test_split(images,
                                                            image_masks,
                                                            test_size=0.25,
                                                            random_state=config.RANDOM_SEED,
                                                            shuffle=True)
        
    dataset_train = create_dataset_cropped(X_train, y_train, batch_sizes[0])
    dataset_val = create_dataset_cropped(X_val, y_val, batch_sizes[0])
    
    model = full_training(model, dataset_train, dataset_val, 70, 80)
    model.save("../saved_models/segmentation_model-upsizing_checkpoint_{}.h5".format(str(0)))
    
    
    for i in range(2, len(sizes)):
        if i==(len(sizes)-1):
            config.patches == "full"
        config.VGG_IMG_SIZE["HEIGHT"] = sizes[i][0]
        config.VGG_IMG_SIZE["WIDTH"] = sizes[i][1]
    
        model = u_net_res_model(input_height = config.VGG_IMG_SIZE["HEIGHT"], input_width = config.VGG_IMG_SIZE["WIDTH"])
        model.load_weights("../saved_models/segmentation_model-upsizing_checkpoint_{}.h5".format(str(i-1)))
        
        # So samples in train and validation still remain in same split but are shuffled
        X_train, X_val, y_train, y_val = train_test_split(images,
                                                            image_masks,
                                                            test_size=0.25,
                                                            random_state=config.RANDOM_SEED,
                                                            shuffle=True)
        
        if i==(len(sizes)-1):
            dataset_train = create_dataset_masks(X_train, y_train)
            dataset_val = create_dataset_masks(X_val, y_val)
        else:
            dataset_train = create_dataset_cropped(X_train, y_train, batch_sizes[i])
            dataset_val = create_dataset_cropped(X_val, y_val, batch_sizes[i])

        model = partial_training(model, dataset_train, dataset_val, 80)
        model.save("../saved_models/segmentation_model-upsizing_checkpoint_{}.h5".format(str(i)))
        
        y_pred = make_predictions(model, dataset_val)
        
        if i==(len(sizes)-1):
            evaluate_segmentation(y_val, y_pred, threshold = 0.5)
            visualise_examples(X_val, y_val, y_pred, threshold = 0.5)

    return model


def full_training(model, train_data, val_data, epochs1, epochs2):
    if config.pretrained=="imagenet":
        for i in range(140):
            model.layers[i].trainable = False
    else:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        
    
    model.compile(optimizer=Adam(lr=0.0001),
                      loss=dual_loss_weighted,
                      metrics=[BinaryAccuracy()])

    
    hist_1 = model.fit(x=train_data,
                           validation_data=val_data,
                           epochs=epochs1,
                           callbacks=[
                               EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                               ReduceLROnPlateau(patience=4)]
                           )

    
    # Train a second time with a smaller learning rate and with all layers unfrozen
    # (train over fewer epochs to prevent over-fitting).

    for i in range(len(model.layers)):
        model.layers[i].trainable = True
            
    model.compile(optimizer=Adam(lr=0.00001),
                      loss=dual_loss_weighted,
                      metrics=[BinaryAccuracy()])

    
    hist_2 = model.fit(x=train_data,
                           validation_data=val_data,
                           epochs=epochs2,
                           callbacks=[
                               EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                               ReduceLROnPlateau(patience=6)]
                           )
    
    return model


def partial_training(model, train_data, val_data, epochs):
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
        
    l_r = 0.00005
    
    model.compile(optimizer=Adam(lr=l_r),
                      loss=dual_loss_weighted,
                      metrics=[BinaryAccuracy()])

    
    hist_1 = model.fit(x=train_data,
                           validation_data=val_data,
                           epochs=epochs,
                           callbacks=[
                               EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                               ReduceLROnPlateau(patience=4)]
                           )
    
    return model



def make_predictions(model, x_values):
    """
    :param model: The CNN model.
    :param x: Input.
    :return: Model predictions.
    """

    y_predict = model.predict(x=x_values)
    return y_predict


def dice_loss(y_true, y_pred):
    numerator = 2 * tf.math.reduce_sum(y_true * y_pred, axis=(1,2))
    denominator = tf.math.reduce_sum(y_true + y_pred, axis=(1,2))

    return 1 - numerator / denominator


def weighted_cross_entropy(beta):
    def convert_to_logits(y_pred):
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())

        return tf.math.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=beta)

        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss)

    return loss


def dual_loss(y_true, y_pred):
    def dice_loss(y_true, y_pred):
        numerator = 2 * tf.math.reduce_sum(y_true * y_pred, axis=(1,2))
        denominator = tf.math.reduce_sum(y_true + y_pred, axis=(1,2))

        return tf.reshape(1 - numerator / denominator, (-1, 1, 1))

    return (binary_crossentropy(y_true, y_pred) + (2*dice_loss(y_true, y_pred)))/3


def dual_loss_weighted(y_true, y_pred):
    def dice_loss(y_true, y_pred):
        numerator = 2 * tf.math.reduce_sum(y_true * y_pred, axis=(1,2))
        denominator = tf.math.reduce_sum(y_true + y_pred, axis=(1,2))

        return tf.reshape(1 - numerator / denominator, (-1, 1, 1))
    
    def weighted_cross(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        y_pred = tf.math.log(y_pred / (1 - y_pred))
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=2)
        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss)
    return (0.34*weighted_cross(y_true, y_pred)) + (0.66*dice_loss(y_true, y_pred))