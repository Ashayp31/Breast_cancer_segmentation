from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from model.output import plot_training_results

import config

def train_network(model, train_x, train_y, val_x, val_y, batch_s, epochs1, epochs2):
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
    # Freeze VGG19 pre-trained layers.
    model.layers[0].trainable = False

    # Train model with frozen layers (all training with early stopping dictated by loss in validation over 3 runs).
    model.compile(optimizer=Adam(lr=1e-3),
                  loss=CategoricalCrossentropy(),
                  metrics=[CategoricalAccuracy()])

    if config.dataset == "mini-MIAS":
        hist = model.fit(
            x=train_x,
            y=train_y,
            batch_size=batch_s,
            steps_per_epoch=len(train_x) // batch_s,
            validation_data=(val_x, val_y),
            validation_steps=len(val_x) // batch_s,
            epochs=epochs1,
            callbacks=[
                EarlyStopping(monitor='val_categorical_accuracy', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=6)
            ]
        )

    elif config.dataset == "CBIS-DDSM":
        hist_1 = model.fit(x=train_x,
                  validation_data = val_x,
                            epochs = epochs1,
                                     callbacks = [
            EarlyStopping(monitor='val_categorical_accuracy', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=6)]
        )

    # Plot the training loss and accuracy.
    plot_training_results(hist, "Initial_training", True)

    # Train a second time with a smaller learning rate and with all layers unfrozen
    # (train over fewer epochs to prevent over-fitting).
    model.layers[0].trainable = True
    model.compile(optimizer=Adam(1e-5),  # Very low learning rate
                  loss=CategoricalCrossentropy(),
                  metrics=[CategoricalAccuracy()])

    if config.dataset == "mini-MIAS":
        hist_2 = model.fit(
            x=train_x,
            y=train_y,
            batch_size=batch_s,
            steps_per_epoch=len(train_x) // batch_s,
            validation_data=(val_x, val_y),
            validation_steps=len(val_x) // batch_s,
            epochs=epochs2,
            callbacks=[
                EarlyStopping(monitor='val_categorical_accuracy', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=6)
            ]
        )
    elif config.dataset == "CBIS-DDSM":
        hist_2 = model.fit(x=train_x,
                  validation_data=val_x,
                  epochs=epochs2,
                  callbacks=[
                      EarlyStopping(monitor='val_categorical_accuracy', patience=10, restore_best_weights=True),
                      ReduceLROnPlateau(patience=6)]
                  )

    # Plot the training loss and accuracy.
    plot_training_results(hist_2, "Fine_tuning_training", False)

    return model


def make_predictions(model, x):
    """
    :param model: The CNN model.
    :param x: Input.
    :return: Model predictions.
    """
    y_predict = model.predict(x=x.astype("float32"), batch_size=10)
    return y_predict
