import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
from skimage import exposure
from scipy.ndimage import gaussian_filter
from PIL import Image 
from tensorflow.keras import backend as K

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
#     as_png = tf.image.encode_png(image[0])
#     decoded_png = tf.io.decode_png(as_png, channels=1)
    image = tf.image.resize(decoded_png, [config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH']])
    current_min = tf.reduce_min(image)
    current_max = tf.reduce_max(image)
    image = (image - current_min) / (current_max - current_min)
    return image, label


def parse_function_large(filename, label):
    """
    mapping function to convert filename to array of pixel values for larger images we use resize with padding
    """
    image_bytes = tf.io.read_file(filename)
    image = tfio.image.decode_dicom_image(image_bytes,color_dim = True,  dtype=tf.uint16)
#     as_png = tf.image.encode_png(image[0])
#     decoded_png = tf.io.decode_png(as_png, channels=1)
    image = tf.image.resize_with_pad(decoded_png, config.VGG_IMG_SIZE_LARGE['HEIGHT'], config.VGG_IMG_SIZE_LARGE['WIDTH'])
    current_min = tf.reduce_min(image)
    current_max = tf.reduce_max(image)
    image = (image - current_min) / (current_max - current_min)
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
    """
    Mapping function for tensorflow dataset for giving ground truth masks and input images
    """
    image_bytes = tf.io.read_file(filename_original)
    image = tfio.image.decode_dicom_image(image_bytes, color_dim = True)
    image = tf.image.resize_with_pad(image, config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'])
    if config.prep == "Y":
        image /= 256
        image = gaussian_blur(image)
        image = tf_equalize_histogram(image)
    current_min = tf.reduce_min(image)
    current_max = tf.reduce_max(image)
    image = (image - current_min) / (current_max - current_min)
    image = tf.reshape(image, [config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'],1])


    image_bytes_mask = tf.io.read_file(filename_mask)
    image_mask = tfio.image.decode_dicom_image(image_bytes_mask, color_dim = True)
    image_mask = tf.image.resize_with_pad(image_mask[0], config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'])
    current_min_mask = tf.reduce_min(image_mask)
    current_max_mask = tf.reduce_max(image_mask)
    image_mask = (image_mask - current_min_mask) / (current_max_mask - current_min_mask)
    image_mask = tf.reshape(image_mask, [-1, 1])

    return image, image_mask


def equalise_and_filter(im):
    """
    Function for applying gaussian filter and histogram equalisation of images
    """
    im_array = tf.make_ndarray(im)
    img_eq = exposure.equalize_hist(im_array)
    gaussian_filtered=gaussian_filter(img_eq, sigma=0.5)
    as_tensor = tf.convert_to_tensor(gaussian_filtered)
    return as_tensor
    

def crop_image_from_bbox(im, im_mask, bbox_model):
    """
    Function to crop an image from a bounding box
    """
    im_array = np.array(im)
    as_image = Image.fromarray(im_array)
    
    im_array_mask = np.array(im_mask)
    as_image_mask = Image.fromarray(im_array_mask)

    # bbox = bbox_model.predict(im)
    (left, top, right, bottom) = bbox
    
    image_crop = as_image.crop((left, top, right, bottom)) 
    mask_crop = as_image_mask.crop((left, top, right, bottom))
    
    image_as_array = numpy.array(image_crop)
    mask_as_array = numpy.array(mask_crop)

    image_as_tensor = tf.convert_to_tensor(image_as_array)
    mask_as_tensor = tf.convert_to_tensor(mask_as_array)

    return image_as_tensor, mask_as_tensor


def tf_equalize_histogram(image):
    """ 
    Function to undertake histogram equalisation 
    """
    values_range = tf.constant([0., 255.], dtype = tf.float32)
    histogram = tf.histogram_fixed_width(tf.cast(image, tf.float32), values_range, 256)
    cdf = tf.cumsum(histogram)
    cdf_min = cdf[tf.reduce_min(tf.where(tf.greater(cdf, 0)))]

    img_shape = tf.shape(image)
    pix_cnt = img_shape[-3] * img_shape[-2]
    px_map = tf.round(tf.cast(cdf - cdf_min, tf.float32) * 255. / tf.cast(pix_cnt - 1, tf.float32))
    px_map = tf.cast(px_map, tf.uint8)
    gathered = tf.gather_nd(px_map, tf.cast(image, tf.int32))
    eq_hist = tf.expand_dims(gathered, 3)
    return eq_hist


def gaussian_blur(img, kernel_size=11, sigma=0.5):
    """ 
    Function for gaussian filtering
    """
    def gauss_kernel(channels, kernel_size, sigma):
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel

    gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
    gaussian_kernel = gaussian_kernel[..., tf.newaxis]

    return tf.nn.depthwise_conv2d(img, gaussian_kernel, [1, 1, 1, 1],
                                  padding='SAME', data_format='NHWC')
