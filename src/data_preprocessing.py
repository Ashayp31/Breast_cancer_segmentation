import os

from imutils import paths
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical

VGG_IMG_HEIGHT = 224
VGG_IMG_WIDTH = 224


def import_dataset(data_dir: str) -> (np.ndarray, np.ndarray):
    """
    Import the dataset by pre-processing the images and encoding the labels.
    :return: Two NumPy arrays, one for processed images and one for the encoded labels.
    """
    # Initialise variables.
    dataset = list()
    labels = list()

    # Loop over the image paths and update the data and labels lists with the pre-processed images & labels.
    for image_path in list(paths.list_images(data_dir)):
        dataset.append(preprocess_image(image_path))
        labels.append(image_path.split(os.path.sep)[-2])  # Extract label from path.

    # Convert the data and labels lists to NumPy arrays.
    dataset = np.array(dataset, dtype="float32")  # Convert images to a batch.
    labels = np.array(labels)

    # Encode labels.
    labels = encode_labels(labels)

    return dataset, labels


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Pre-processing steps:
        * Load the input image in grayscale mode (1 channel),
        * resize it to 224x224 pixels for the VGG19 CNN model,
        * transform it to an array format.
    :param image_path: The path to the image to preprocess.
    :return: The pre-processed image in NumPy array format.
    """
    image = load_img(image_path, color_mode="grayscale", target_size=(VGG_IMG_HEIGHT, VGG_IMG_WIDTH))
    image = img_to_array(image)
    return image


def encode_labels(labels_list: np.ndarray) -> np.ndarray:
    """
    Encode labels using one-hot encoding.
    :param labels_list: The list of labels in NumPy array format.
    :return: The encoded list of labels in NumPy array format.
    """
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels_list)
    return to_categorical(labels)


def train_test_split_dataset(dataset: np.ndarray, labels: np.ndarray) -> \
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Partition the data into training and testing splits using 80%/20% split. Stratify the split to keep the same class
    distribution in both sets.
    :param dataset: The dataset of pre-processed images.
    :param labels: The list of labels.
    :return: the training and testing sets split in input (X) and label (Y).
    """
    train_X, test_X, train_Y, test_Y = train_test_split(dataset,
                                                        labels,
                                                        test_size=0.20,
                                                        stratify=labels,
                                                        random_state=111)
    return train_X, test_X, train_Y, test_Y
