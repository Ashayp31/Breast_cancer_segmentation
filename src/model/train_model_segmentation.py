from tensorflow.keras.losses import BinaryCrossentropy
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
        * Train network with initial VGG base layers frozen
        * Unfreeze all layers and retrain with smaller learning rate
    :param model: CNN model
    :param train_x: training input
    :param train_y: training outputs
    :param val_x: validation inputs
    :param val_y: validation outputs
    :param batch_s: batch size
    :param epochs1: epoch count for initial training
    :param epochs2: epoch count for training all layers unfrozen
    :return: trained network
    """
    # Assuming resnet 50
    for i in range(140):
        model.layers[i].trainable = False

    # Train model with frozen layers (all training with early stopping dictated by loss in validation over 3 runs).


#     model.compile(optimizer=Adam(lr=1e-4),
#                       loss=BinaryCrossentropy(),
#                       metrics=[BinaryAccuracy()])
    model.compile(optimizer=Adam(lr=1e-3),
                      loss=dice_coef_loss,
                      metrics=[BinaryAccuracy()])

    hist_1 = model.fit(x=train_x,
                           validation_data=val_x,
                           epochs=epochs1,
                           callbacks=[
                               EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                               ReduceLROnPlateau(patience=4)]
                           )

    
#     print(hist_1.history)
    # Plot the training loss and accuracy.
    plot_training_results_segmentation(hist_1, "Initial_training_segmentation", True)

    # Train a second time with a smaller learning rate and with all layers unfrozen
    # (train over fewer epochs to prevent over-fitting).
    for i in range(len(model.layers)):
        model.layers[i].trainable = True

    model.compile(optimizer=Adam(lr=1e-5),
                      loss=dice_coef_loss,
                      metrics=[BinaryAccuracy()])

    hist_2 = model.fit(x=train_x,
                           validation_data=val_x,
                           epochs=epochs2,
                           callbacks=[
                               EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
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


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """

    y_true = y_true[0, :, 0]
    y_pred = y_pred[0, :, 0]
    intersection = K.sum(K.abs(y_true * y_pred))
    return (2. * intersection + smooth) / (K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)