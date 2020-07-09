import tensorflow as tf
import tensorflow_io as tfio

import config


def create_dataset(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    # map values from dicom image path to array
    if config.image == "small":
        dataset = dataset.map(parse_function_small, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(parse_function_large, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache().repeat(1)
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def parse_function_small(filename, label):
    """

    :param filename:
    :param label:
    :return:
    """
    image_bytes = tf.io.read_file(filename)
    image = tfio.image.decode_dicom_image(image_bytes, color_dim=True, scale="auto", dtype=tf.uint16)
    as_png = tf.image.encode_png(image[0])
    decoded_png = tf.io.decode_png(as_png, channels=1)
    image = tf.image.resize(decoded_png, [config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['HEIGHT']])
    image /= 255

    return image, label


def parse_function_large(filename, label):
    image_bytes = tf.io.read_file(filename)
    image = tfio.image.decode_dicom_image(image_bytes,color_dim = True,  dtype=tf.uint16)
    as_png = tf.image.encode_png(image[0])
    decoded_png = tf.io.decode_png(as_png, channels=1)
    image = tf.image.resize_with_pad(decoded_png, config.VGG_IMG_SIZE_LARGE['HEIGHT'], config.VGG_IMG_SIZE_LARGE['HEIGHT'])
    image /= 255

    return image, label
