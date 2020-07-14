import tensorflow as tf
import tensorflow_io as tfio
import numpy as np

import config


def create_dataset(x, y):
    """
    Genereate a tensorflow dataset for feeding in the data
    :param x: X inputs - paths to images
    :param y: y values - labels for images
    :return: the dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    # map values from dicom image path to array
    if config.imagesize == "small":
        dataset = dataset.map(parse_function_small, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(parse_function_large, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Dataset to cache data and repeat until all samples have been run once in each epoch
    dataset = dataset.cache().repeat(1)
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def parse_function_small(filename, label):
    """
    mapping function to convert filename to array of pixel values
    :param filename:
    :param label:
    :return:
    """
    image_bytes = tf.io.read_file(filename)
    image = tfio.image.decode_dicom_image(image_bytes, color_dim=True, scale="auto", dtype=tf.uint16)
    as_png = tf.image.encode_png(image[0])
    decoded_png = tf.io.decode_png(as_png, channels=1)
    image = tf.image.resize(decoded_png, [config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH']])
    image /= 255

    return image, label


def parse_function_large(filename, label):
    """
    mapping function to convert filename to array of pixel values for larger images we use resize with padding
    """
    image_bytes = tf.io.read_file(filename)
    image = tfio.image.decode_dicom_image(image_bytes,color_dim = True,  dtype=tf.uint16)
    as_png = tf.image.encode_png(image[0])
    decoded_png = tf.io.decode_png(as_png, channels=1)
    image = tf.image.resize_with_pad(decoded_png, config.VGG_IMG_SIZE_LARGE['HEIGHT'], config.VGG_IMG_SIZE_LARGE['WIDTH'])
    image /= 255

    return image, label


def create_dataset_masks(x, y):
    """
    Genereate a tensorflow dataset for feeding in the data
    :param x: X inputs - paths to images
    :param y: y values - labels for images
    :return: the dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(parse_function_segmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Dataset to cache data and repeat until all samples have been run once in each epoch
    dataset = dataset.cache().repeat(1)
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def parse_function_segmentation(filename_original, filename_mask):
    image_bytes = tf.io.read_file(filename_original)
    image = tfio.image.decode_dicom_image(image_bytes, color_dim = True)
    image = tf.image.resize_with_pad(image[0], config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'])
    current_min = tf.reduce_min(image)
    current_max = tf.reduce_max(image)
    image = (image - current_min) / (current_max - current_min)


    image_bytes_mask = tf.io.read_file(filename_mask)
    image_mask = tfio.image.decode_dicom_image(image_bytes_mask, color_dim = True)
    image_mask = tf.image.resize_with_pad(image_mask[0], config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'])
    current_min_mask = tf.reduce_min(image_mask)
    current_max_mask = tf.reduce_max(image_mask)
    image_mask = (image - current_min_mask) / (current_max_mask - current_min_mask)
    image_mask = tf.reshape(image_mask, [-1, 1])

    return image, image_mask
