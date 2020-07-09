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
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache().repeat(1)
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def parse_function(filename, label):
    """

    :param filename:
    :param label:
    :return:
    """
    image_bytes = tf.io.read_file(filename)
    image = tfio.image.decode_dicom_image(image_bytes, color_dim=True, scale="auto", dtype=tf.uint16)
    as_png = tf.image.encode_png(image[0])
    decoded_png = tf.io.decode_png(as_png, channels=1)
    image = tf.image.resize(decoded_png, [512, 512])
    image /= 255

    return image, label
