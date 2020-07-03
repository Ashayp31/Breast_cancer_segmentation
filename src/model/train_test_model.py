import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy

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

    # Plot the training loss and accuracy.
    plot_training_results(hist, "Initial_training")

    # Train a second time with a smaller learning rate and with all layers unfrozen
    # (train over fewer epochs to prevent over-fitting).
    model.layers[0].trainable = True
    model.compile(optimizer=Adam(1e-5),  # Very low learning rate
                  loss=CategoricalCrossentropy(),
                  metrics=[CategoricalAccuracy()])
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

    # Plot the training loss and accuracy.
    plot_training_results(hist_2, "Fine_tuning_training")

    return model


def test_network(model, test_x):
    """
    :param model:  CNN model
    :param test_x: testing inputs
    :return: ouputs predicted by the model
    """
    y_predict = model.predict(x=test_x.astype("float32"), batch_size=10)

    return y_predict


def plot_training_results(hist_input, plot_name: str) -> None:
    """
    Function to plot loss and accuracy over epoch count for training
    :param hist_input: The training history.
    :param plot_name: The plot name.
    """
    n = len(hist_input.history["loss"])
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, n), hist_input.history["loss"], label="train_loss")
    plt.plot(np.arange(0, n), hist_input.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, n), hist_input.history["categorical_accuracy"], label="train_acc")
    plt.plot(np.arange(0, n), hist_input.history["val_categorical_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset with all layers unfrozen")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("../output/{}_{}-model_{}-dataset.png".format(plot_name, config.model, config.dataset))
    plt.show()
