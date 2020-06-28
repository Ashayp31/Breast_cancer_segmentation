from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
import matplotlib.pyplot as plt
import numpy as np

def train_network(model, train_gen, val_gen, train_x, train_y, val_x, val_y, batch_s, epochs1, epochs2):
    # Train model with frozen layers
    # ALl training with early stopping dictated by loss in validation over 3 runs
    model.compile(optimizer=Adam(lr=1e-3),
                  loss=CategoricalCrossentropy(),
                  metrics=[CategoricalAccuracy()])

    Hist = model.fit(
        # x=train_gen.flow(train_x, train_y, batch_size=batch_size),
        x= train_x,
        y=train_y,
        batch_size = batch_s,
        steps_per_epoch=len(train_x) // batch_s,
        # validation_data=val_gen.flow(val_x, val_y),
        validation_data=(val_x, val_y),
        validation_steps=len(val_x) // batch_s,
        epochs=epochs1,
        callbacks=[
            EarlyStopping(patience=4, restore_best_weights=True),
            ReduceLROnPlateau(patience=4)
        ]
    )

    # Plot validation and train loss over course of training

    # plot the training loss and accuracy
    plot_training_results(Hist, "Initial_training.png")

    # Train again with slower learning rate unfreezing all layers
    # Train over fewer epochs to stop overfitting
    model.layers[0].trainable = True

    model.compile(optimizer=Adam(1e-5),  # Very low learning rate
                  loss=CategoricalCrossentropy(),
                  metrics=[CategoricalAccuracy()])

    Hist_2 = model.fit(
        x=train_x,
        y=train_y,
	batch_size = batch_s,
        steps_per_epoch=len(train_x) // batch_s,
        validation_data=(val_x, val_y),
        validation_steps=len(val_x) // batch_s,
        epochs=epochs2,
        callbacks=[
            EarlyStopping(patience=4, restore_best_weights=True),
            ReduceLROnPlateau(patience=4)
        ]
    )

    plot_training_results(Hist_2, "Fine_tuning_training.png")

    return model


def test_network(model, test_x):
    # run predictions on text images and return predictions
    y_predict = model.predict(x=test_x.astype("float32"), batch_size=10)

    return y_predict


def plot_training_results(hist, plot_name):
    # plot the training loss and accuracy
    N = len(hist.history["loss"])
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), hist.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), hist.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), hist.history["categorical_accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), hist.history["val_categorical_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset with all layers unfrozen")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plot_name)
    plt.show()

