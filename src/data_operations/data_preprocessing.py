"""
Preliminary code created by Adam Jaamour for data preprocessing of minimias and cbis-ddsm dataset
Code adapted and altered by Me for use for own specific tasks
"""

import os
import random

from imutils import paths
import numpy as np
import pandas as pd
import skimage as sk
import skimage.transform
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical

import config


def import_minimias_dataset(data_dir: str, label_encoder) -> (np.ndarray, np.ndarray):
    """
    Import the dataset by pre-processing the images and encoding the labels.
    :param data_dir: Directory to the mini-MIAS images.
    :param label_encoder: The label encoder.
    :return: Two NumPy arrays, one for the processed images and one for the encoded labels.
    """
    # Initialise variables.
    images = list()
    labels = list()

    # Loop over the image paths and update the data and labels lists with the pre-processed images & labels.
    for image_path in list(paths.list_images(data_dir)):
        images.append(preprocess_image(image_path))
        labels.append(image_path.split(os.path.sep)[-2])  # Extract label from path.

    # Convert the data and labels lists to NumPy arrays.
    images = np.array(images, dtype="float32")  # Convert images to a batch.
    labels = np.array(labels)

    # Encode labels.
    labels = encode_labels(labels, label_encoder)

    return images, labels


def import_cbisddsm_training_dataset(label_encoder):
    """
    Import the dataset getting the image paths (downloaded on BigTMP) and encoding the labels.
    :param label_encoder: The label encoder.
    :return: Two arrays, one for the image paths and one for the encoded labels.
    """
    df = pd.read_csv("../data/CBIS-DDSM/training.csv")
    list_IDs = df['img_path'].values
    labels = encode_labels(df['label'].values, label_encoder)
    return list_IDs, labels


def import_cbisddsm_testing_dataset(label_encoder):
    """
    Import the dataset getting the image paths (downloaded on BigTMP) and encoding the labels.
    :param label_encoder: The label encoder.
    :return: Two arrays, one for the image paths and one for the encoded labels.
    """
    df = pd.read_csv("../data/CBIS-DDSM/testing.csv") 
    list_IDs = df['img_path'].values
    labels = encode_labels(df['label'].values, label_encoder)
    return list_IDs, labels


def import_cbisddsm_mask_classification_training_dataset(label_encoder):
    """
    Import the dataset getting the image paths (downloaded on BigTMP) and encoding the labels.
    :param label_encoder: The label encoder.
    :return: Two arrays, one for the image paths and one for the encoded labels.
    """
    df = pd.read_csv("../data/CBIS-DDSM-mask/final_mask_training.csv")
    list_IDs = df['mask_img_path'].values
    labels = encode_labels(df['label'].values, label_encoder)
    return list_IDs, labels


def import_cbisddsm_segmentation_training_dataset():
    """
    Import the dataset getting the image paths (downloaded on BigTMP) for images and masks.
    :param label_encoder: The label encoder.
    :return: Two arrays, one for the image paths and one for the encoded labels.
    """
#     df = pd.read_csv("../data/CBIS-DDSM-mask/final_mask_training_short.csv")

    df = pd.read_csv("../data/CBIS-DDSM-mask/final_mask_training.csv")

    list_IDs = df['img_path'].values
    ground_truth_ids = df['mask_img_path'].values
    return list_IDs, ground_truth_ids


def import_cbisddsm_segmentation_testing_dataset():
    """
    Import the dataset getting the image paths (downloaded on BigTMP) for images and masks.
    :param label_encoder: The label encoder.
    :return: Two arrays, one for the image paths and one for the encoded labels.
    """

    df = pd.read_csv("../data/CBIS-DDSM-mask/final_mask_testing.csv")
#     df = pd.read_csv("../data/CBIS-DDSM-mask/final_mask_training-Copy1.csv")

    list_IDs = df['img_path'].values
    ground_truth_ids = df['mask_img_path'].values
    return list_IDs, ground_truth_ids


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Pre-processing steps:
        * Load the input image in grayscale mode (1 channel),
        * resize it to 224x224 pixels for the VGG19 CNN model,
        * transform it to an array format,
        * normalise the pixel intensities.
    :param image_path: The path to the image to preprocess.
    :return: The pre-processed image in NumPy array format.
    """
    image = load_img(image_path,
                     color_mode="grayscale",
                     target_size=(config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE["WIDTH"]))
    image = img_to_array(image)
    image /= 255.0
    return image


def encode_labels(labels_list: np.ndarray, label_encoder) -> np.ndarray:
    """
    Encode labels using one-hot encoding.
    :param label_encoder: The label encoder.
    :param labels_list: The list of labels in NumPy array format.
    :return: The encoded list of labels in NumPy array format.
    """
    labels = label_encoder.fit_transform(labels_list)
    if label_encoder.classes_.size == 2:
        return labels
    else:
        return to_categorical(labels)


def dataset_stratified_split(split: float, dataset: np.ndarray, labels: np.ndarray) -> \
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Partition the data into training and testing splits. Stratify the split to keep the same class distribution in both
    sets and shuffle the order to avoid having imbalanced splits.
    :param split: Dataset split (e.g. if 0.2 is passed, then the dataset is split in 80%/20%).
    :param dataset: The dataset of pre-processed images.
    :param labels: The list of labels.
    :return: the training and testing sets split in input (X) and label (Y).
    """
    train_X, test_X, train_Y, test_Y = train_test_split(dataset,
                                                        labels,
                                                        test_size=split,
                                                        stratify=labels,
                                                        random_state=config.RANDOM_SEED,
                                                        shuffle=True)
    return train_X, test_X, train_Y, test_Y


def random_rotation(image_array: np.ndarray):
    """
    Randomly rotate the image
    :param image_array: input image
    :return: randomly rotated image
    """
    random_degree = random.uniform(-20, 20)
    return sk.transform.rotate(image_array, random_degree)


def random_noise(image_array: np.ndarray):
    """
    Add random noise to image
    :param image_array: input image
    :return: image with added random noise
    """
    return sk.util.random_noise(image_array)


def horizontal_flip(image_array: np.ndarray):
    """
    Flip image
    :param image_array: input image
    :return: horizantally flipped image
    """
    return image_array[:, ::-1]


def generate_image_transforms(images, labels):
    """
    oversample data by tranforming existing images
    :param images: input images
    :param labels: input labels
    :return: updated list of images and labels with extra transformed images and labels
    """
    images_with_transforms = images
    labels_with_transforms = labels

    # Possible tranformations to do
    available_transforms = {'rotate': random_rotation,
                            'noise': random_noise,
                            'horizontal_flip': horizontal_flip}

    # Get count of each class and find out how many needs to be added to each class so that they have the same number
    # Class with highest count will have 0 to add
    class_balance = get_class_balances(labels)
    max_count = max(class_balance)
    to_add = [max_count - i for i in class_balance]

    # For each type of class (2 binary, 3 mini mias)
    for i in range(len(to_add)):
        if int(to_add[i]) == 0:
            continue
        # Generate label of class
        label = np.zeros(len(to_add))
        label[i] = 1
        # Get indices in data of samples of this class
        indices = [j for j, x in enumerate(labels) if np.array_equal(x, label)]
        indiv_class_images = [images[j] for j in indices]

        # For the number of samples to add as calculated before
        for k in range(int(to_add[i])):
            # Generate a a transoformed image from a sample in the class
            transformed_image = create_individual_transform(indiv_class_images[k % len(indiv_class_images)],
                                                            available_transforms)
            transformed_image = transformed_image.reshape(1, config.VGG_IMG_SIZE['HEIGHT'],
                                                          config.VGG_IMG_SIZE['WIDTH'], 1)
            # Append transormed images and labels to current list
            images_with_transforms = np.append(images_with_transforms, transformed_image, axis=0)
            transformed_label = label.reshape(1, len(label))
            labels_with_transforms = np.append(labels_with_transforms, transformed_label, axis=0)

    return images_with_transforms, labels_with_transforms


def create_individual_transform(image: np.array, transforms: dict):
    """
    Create transformation of an individual image
    :param image: input image
    :param transforms: the possible transforms to do on the image
    :return: transformed image
    """
    num_transformations_to_apply = random.randint(1, len(transforms))
    num_transforms = 0
    transformed_image = None
    while num_transforms <= num_transformations_to_apply:
        key = random.choice(list(transforms))
        transformed_image = transforms[key](image)
        num_transforms += 1

    return transformed_image


def get_class_balances(y_vals):
    """
    Count occurrences of each class.
    :param y_vals: labels
    :return: array count of each class
    """
    num_classes = len(y_vals[0])
    counts = np.zeros(num_classes)
    for y_val in y_vals:
        for i in range(num_classes):
            counts[i] += y_val[i]

    return (counts.tolist())
