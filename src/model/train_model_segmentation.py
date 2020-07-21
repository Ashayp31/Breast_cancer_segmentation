from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError, binary_crossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import config
from data_visualisation.output import plot_training_results_segmentation
from tensorflow.keras import backend as K



def train_segmentation_network(model, train_x, val_x, epochs1, epochs2):
    """
    Function to train network in two steps:
        * Train network with initial ResNET backbone layers frozen
        * Unfreeze all layers and retrain with smaller learning rate
    """
    l_r = 0.001
    
    # Assuming resnet 50
    if config.pretrained == "imagenet":
        for i in range(140):
            model.layers[i].trainable = False
    elif config.patches == "Y":
        l_r = 0.0001

    # Train model with frozen layers (all training with early stopping dictated by loss in validation over 3 runs).


    model.compile(optimizer=Adam(lr=l_r),
                      loss=dual_loss_weighted,
                      metrics=[BinaryAccuracy()])

    hist_1 = model.fit(x=train_x,
                           validation_data=val_x,
                           epochs=epochs1,
                           callbacks=[
                               EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                               ReduceLROnPlateau(patience=4)]
                           )

    
    # Plot the training loss and accuracy.
    plot_training_results_segmentation(hist_1, "Initial_training_segmentation", True)

    # Train a second time with a smaller learning rate and with all layers unfrozen
    # (train over fewer epochs to prevent over-fitting).

    if config.pretrained == "imagenet":
        for i in range(len(model.layers)):
            model.layers[i].trainable = True

    model.compile(optimizer=Adam(lr=0.00005),
                      loss=dual_loss_weighted,
                      metrics=[BinaryAccuracy()])

    hist_2 = model.fit(x=train_x,
                           validation_data=val_x,
                           epochs=epochs2,
                           callbacks=[
                               EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                               ReduceLROnPlateau(patience=6)]
                           )

    
    # Plot the training loss and accuracy.
    plot_training_results_segmentation(hist_2, "Fine_tuning_training_segmentation", False)

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

    return (0.5*binary_crossentropy(y_true, y_pred)) + (0.5*dice_loss(y_true, y_pred))


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
    return (0.5*weighted_cross(y_true, y_pred)) + (0.5*dice_loss(y_true, y_pred))