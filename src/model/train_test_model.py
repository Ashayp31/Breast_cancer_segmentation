from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from model.output import plot_training_results


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

    # Train model with frozen layers
    # ALl training with early stopping dictated by loss in validation over 3 runs
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

    # plot the training loss and accuracy
    plot_training_results(hist, "Initial_training", True)

    # Train again with slower learning rate unfreezing all layers
    # Train over fewer epochs to stop overfitting
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
<<<<<<< HEAD


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
    plt.title("Training Loss and Accuracy Evolution on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("../output/{}_{}-model_{}-dataset.png".format(plot_name, config.model, config.dataset))
    plt.show()
=======
>>>>>>> e5351536e96ba4aeeb7572b45b7fd96a48cd76d8
