import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
from skimage import exposure
from scipy.ndimage import gaussian_filter
from PIL import Image 
from tensorflow.keras import backend as K
import pydicom
from skimage.measure import label, regionprops


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
    image = tf.image.resize_with_pad(image, config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'])
    if config.prep == "Y":
        image /= 256
        image = gaussian_blur(image)
        image = tf_equalize_histogram(image)
    current_min = tf.reduce_min(image)
    current_max = tf.reduce_max(image)
    image = (image - current_min) / (current_max - current_min)
    image = tf.reshape(image, [config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'],1])
    return image, label


def parse_function_large(filename, label):
    """
    mapping function to convert filename to array of pixel values for larger images we use resize with padding
    """
    image_bytes = tf.io.read_file(filename)
    image = tfio.image.decode_dicom_image(image_bytes,color_dim = True,  dtype=tf.uint16)
    image = tf.image.resize_with_pad(image, config.VGG_IMG_SIZE_LARGE['HEIGHT'], config.VGG_IMG_SIZE_LARGE['WIDTH'])
    if config.prep == "Y":
        image /= 256
        image = gaussian_blur(image)
        image = tf_equalize_histogram(image)
    current_min = tf.reduce_min(image)
    current_max = tf.reduce_max(image)
    image = (image - current_min) / (current_max - current_min)
    image = tf.reshape(image, [config.VGG_IMG_SIZE_LARGE['HEIGHT'], config.VGG_IMG_SIZE_LARGE['WIDTH'],1])
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
    
    # If config is set for preprocessing we apply gaussian featuring and histogram equalisation
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


def create_dataset_patched(x, y):
    """
    Genereate a tensorflow dataset for feeding in the data
    :param x: X inputs - paths to images
    :param y: y values - labels for images
    :return: the dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(parse_function_patch_segmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.flat_map(lambda x,y: tf.data.Dataset.from_tensor_slices((x,y)))

    # Filter out patches that are all black
    dataset = dataset.filter(filter_fn)

    # Dataset to cache data and repeat until all samples have been run once in each epoch
    dataset = dataset.cache().repeat(1)
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def filter_fn(x,y):
#     return tf.math.greater(tf.reduce_sum(x), 0)
    return tf.math.greater(tf.reduce_sum(y), 0)



def parse_function_patch_segmentation(filename_original, filename_mask):
   
    """
    Mapping function for tensorflow dataset for giving ground truth masks and input images and dividing an image intp
    patches with a stride of half the size of the patches
    """
    image_bytes = tf.io.read_file(filename_original)
    image = tfio.image.decode_dicom_image(image_bytes, color_dim = True)
    image = tf.image.resize_with_pad(image, config.VGG_IMG_SIZE_LARGE['HEIGHT'], config.VGG_IMG_SIZE_LARGE['WIDTH'])
    
    # If config is set for preprocessing we apply gaussian featuring and histogram equalisation
    if config.prep == "Y":
        image /= 256
        image = gaussian_blur(image)
        image = tf_equalize_histogram(image)
    current_min = tf.reduce_min(image)
    current_max = tf.reduce_max(image)
    image = (image - current_min) / (current_max - current_min)
    image = tf.reshape(image, [1, config.VGG_IMG_SIZE_LARGE['HEIGHT'], config.VGG_IMG_SIZE_LARGE['WIDTH'],1])


    image_bytes_mask = tf.io.read_file(filename_mask)
    image_mask = tfio.image.decode_dicom_image(image_bytes_mask, color_dim = True)
    image_mask = tf.image.resize_with_pad(image_mask[0], config.VGG_IMG_SIZE_LARGE['HEIGHT'], config.VGG_IMG_SIZE_LARGE['WIDTH'])
    current_min_mask = tf.reduce_min(image_mask)
    current_max_mask = tf.reduce_max(image_mask)
    image_mask = (image_mask - current_min_mask) / (current_max_mask - current_min_mask)
    image_mask = tf.reshape(image, [1, config.VGG_IMG_SIZE_LARGE['HEIGHT'], config.VGG_IMG_SIZE_LARGE['WIDTH'],1])
    
    patch_size = [config.VGG_IMG_SIZE["HEIGHT"],config.VGG_IMG_SIZE["WIDTH"]] 
    
    # Patch image into a size of patch size as given by the config with a stride equal to half the patch size to
    # give an overlap between each patch

    image_patch = tf.image.extract_patches(images=image,
                           sizes=[1, patch_size[0], patch_size[1], 1],
                           strides=[1, patch_size[0]//2, patch_size[1]//2, 1],
                           rates=[1, 1, 1, 1],
                           padding='SAME')
    
    mask_patch = tf.image.extract_patches(images=image_mask,
                           sizes=[1, patch_size[0], patch_size[1], 1],
                           strides=[1, patch_size[0]//2, patch_size[1]//2, 1],
                           rates=[1, 1, 1, 1],
                           padding='SAME')

    image_patch_reshaped = tf.reshape(image_patch,[100,config.VGG_IMG_SIZE["HEIGHT"],config.VGG_IMG_SIZE["WIDTH"],1])
    mask_patch_reshaped = tf.reshape(mask_patch,[100,-1,1])

    return image_patch_reshaped, mask_patch_reshaped



def create_dataset_cropped(x, y, batch_size):
    """
    Genereate a tensorflow dataset for feeding in the data
    :param x: X inputs - paths to images
    :param y: y values - path for image masks
    :return: the dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(parse_function_crops, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Dataset to cache data and repeat until all samples have been run once in each epoch
    dataset = dataset.cache().repeat(1)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset



def parse_function_crops(filename_original,filename_mask):
    image_bytes = tf.io.read_file(filename_original)
    image = tfio.image.decode_dicom_image(image_bytes, color_dim = True)
    image = tf.image.resize_with_pad(image, config.VGG_IMG_SIZE_LARGE['HEIGHT'], config.VGG_IMG_SIZE_LARGE['WIDTH'])
    if config.prep == "Y":
        image /= 256
        image = gaussian_blur(image)
        image = tf_equalize_histogram(image)
    current_min = tf.reduce_min(image)
    current_max = tf.reduce_max(image)
    image = (image - current_min) / (current_max - current_min)
    image = tf.reshape(image, [1, config.VGG_IMG_SIZE_LARGE['HEIGHT'], config.VGG_IMG_SIZE_LARGE['WIDTH'],1])

    
    
    image_bytes_mask = tf.io.read_file(filename_mask)
    image_mask = tfio.image.decode_dicom_image(image_bytes_mask, color_dim = True)
    image_mask = tf.image.resize_with_pad(image_mask[0], config.VGG_IMG_SIZE_LARGE['HEIGHT'], config.VGG_IMG_SIZE_LARGE['WIDTH'])
    current_min_mask = tf.reduce_min(image_mask)
    current_max_mask = tf.reduce_max(image_mask)
    image_mask = (image_mask - current_min_mask) / (current_max_mask - current_min_mask)
    image_mask = tf.reshape(image_mask, [1, config.VGG_IMG_SIZE_LARGE['HEIGHT'], config.VGG_IMG_SIZE_LARGE['WIDTH'],1])
    
    
    [minr, minc, maxr, maxc] = tf_function(image_mask)
    bbox_height = maxr - minr
    bbox_width = maxc - minc
    
    # from bounding box size to wanted image size get required expansion of bounding box needed
    if bbox_height < config.VGG_IMG_SIZE['HEIGHT']:
        height_to_add = config.VGG_IMG_SIZE['HEIGHT']-bbox_height
    else:
        height_to_add = 0
        
    if bbox_width < config.VGG_IMG_SIZE['WIDTH']:
        width_to_add = config.VGG_IMG_SIZE['WIDTH']-bbox_width
    else:
        width_to_add = 0
        
    if (height_to_add == 0) or (width_to_add == 0):
        image = tf.image.crop_to_bounding_box(image, minr, minc, maxr-minr, maxc-minc)
        image_mask = tf.image.crop_to_bounding_box(image_mask, minr, minc, maxr-minr, maxc-minc) 
    else:
        if width_to_add%2 == 1:
            width_extra = 1
        else:
            width_extra = 0

        if height_to_add%2 == 1:
            height_extra = 1
        else:
            height_extra = 0

        minr = minr - height_to_add//2
        maxr = maxr + height_extra + height_to_add//2
        minc = minc - width_to_add//2
        maxc = maxc + width_extra + width_to_add//2

        # If cropping and resizing takes us outside image bounds we move the crop to the sides
        if minr < 0:
            maxr = maxr + abs(minr)
            minr = 0
        if maxr > config.VGG_IMG_SIZE_LARGE['HEIGHT']:
            minr = minr - (maxr - config.VGG_IMG_SIZE_LARGE['HEIGHT'])
            maxr = config.VGG_IMG_SIZE_LARGE['HEIGHT']

        if minc < 0:
            maxc = maxc + abs(minc)
            minc = 0
        if maxc > config.VGG_IMG_SIZE_LARGE['WIDTH']:
            minc = minc - (maxc - config.VGG_IMG_SIZE_LARGE['WIDTH'])
            maxc = config.VGG_IMG_SIZE_LARGE['WIDTH']


        image = tf.image.crop_to_bounding_box(image, minr, minc, maxr-minr, maxc-minc)
        image_mask = tf.image.crop_to_bounding_box(image_mask, minr, minc, maxr-minr, maxc-minc) 
    
    image = tf.image.resize_with_pad(image, config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'])
    image_mask = tf.image.resize_with_pad(image_mask, config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'])

    image = tf.reshape(image, [config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'],1])
    image_mask = tf.reshape(image_mask, [-1, 1])


    return image, image_mask


@tf.function()
def tf_function(mask_image):
    y = tf.numpy_function(get_bbox_of_mask, [mask_image], [tf.int32, tf.int32, tf.int32, tf.int32])
    return y 

def get_bbox_of_mask(mask):
    mask = mask[0,:,:,0]
    regions = regionprops(mask.astype(int))
    props = regions[0]
    minr, minc, maxr, maxc = props.bbox
    minr = tf.convert_to_tensor(minr, dtype=tf.int32)
    minc = tf.convert_to_tensor(minc, dtype=tf.int32)
    maxr = tf.convert_to_tensor(maxr, dtype=tf.int32)
    maxc = tf.convert_to_tensor(maxc, dtype=tf.int32)

    return [minr, minc, maxr, maxc]





# def bbox_dataset(lines):
#     """
#     Genereate a tensorflow dataset for feeding in the data
#     :param x: X inputs - paths to images
#     :param y: y values - labels for images
#     :return: the dataset
#     """
#     dataset = tf.data.Dataset.from_tensor_slices(lines)
#     dataset = dataset.map(parse_function_bbox, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     # Dataset to cache data and repeat until all samples have been run once in each epoch
#     dataset = dataset.cache().repeat(1)
#     dataset = dataset.batch(config.BATCH_SIZE)
#     dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

#     return dataset


# def parse_function_bbox(line):
#     anchors_path = 'keras_yolo3/model_data/yolo_anchors.txt'
#     anchors = get_anchors(anchors_path)
#     image = get_image(line)
#     box_data = tf_function_1(line)
#     y_true = tf_function_2(box_data, anchors)
#     return image, y_true


# def get_image(annotation_line):
#     """
#     Original code taken from yolo3 implementation code at : https://github.com/qqwweee/keras-yolo3
#     Adapted for CBIS-DDSM dataset input and preprocessing
#     """
#     line = tf.strings.split(annotation_line)

#     image_bytes = tf.io.read_file(line[0])
#     image = tfio.image.decode_dicom_image(image_bytes, color_dim = True,  dtype=tf.uint16)
#     image = tf.image.resize_with_pad(image, config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'])
#     if config.prep == "Y":
#         image /= 256
#         image = gaussian_blur(image)
#         image = tf_equalize_histogram(image)
#     current_min = tf.reduce_min(image)
#     current_max = tf.reduce_max(image)
#     image = (image - current_min) / (current_max - current_min)
#     image_data = tf.reshape(image, [config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'],1])

#     return image_data



# @tf.function(input_signature=[tf.TensorSpec(None, tf.string)])
# def tf_function_1(input):
#     y = tf.numpy_function(my_numpy_func_1, [input], tf.int32)
#     return y 


# def my_numpy_func_1(annotation_line):
#     """
#     Original code taken from yolo3 implementation code at : https://github.com/qqwweee/keras-yolo3
#     Adapted for CBIS-DDSM dataset input and preprocessing
#     """
#     print("WE HERE FOR THIS ONE")
#     line = annotation_line.split()

#     iw, ih = config.VGG_IMG_SIZE["HEIGHT"], config.VGG_IMG_SIZE["WIDTH"] 
#     h, w = config.VGG_IMG_SIZE["HEIGHT"], config.VGG_IMG_SIZE["WIDTH"]
    
#     box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

#     # resize image
#     scale = min(w/iw, h/ih)
#     nw = int(iw*scale)
#     nh = int(ih*scale)
#     dx = (w-nw)//2
#     dy = (h-nh)//2

#     # correct boxes
#     box_data = np.zeros((max_boxes,5))
#     if len(box)>0:
#         np.random.shuffle(box)
#         if len(box)>max_boxes: box = box[:max_boxes]
#         box[:, [0,2]] = box[:, [0,2]]*scale + dx
#         box[:, [1,3]] = box[:, [1,3]]*scale + dy
#         box_data[:len(box)] = box

#     return  box_data



# @tf.function()
# def tf_function_2(boxes, anchors):
#     y = tf.numpy_function(preprocess_true_boxes, [boxes,anchors], tf.int32)
#     return y 


# def preprocess_true_boxes(true_boxes, anchors):

#     num_classes=1,
#     input_shape=(config.VGG_IMG_SIZE["HEIGHT"], config.VGG_IMG_SIZE["WIDTH"])
#     '''
#     Original code taken from yolo3 implementation code at : https://github.com/qqwweee/keras-yolo3
#     Adapted for CBIS-DDSM dataset input and preprocessing
#     Preprocess true boxes to training input format

#     Parameters
#     ----------
#     true_boxes: array, shape=(m, T, 5)
#         Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
#     input_shape: array-like, hw, multiples of 32
#     anchors: array, shape=(N, 2), wh
#     num_classes: integer

#     Returns
#     -------
#     y_true: list of array, shape like yolo_outputs, xywh are reletive value

#     '''
#     assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
#     num_layers = len(anchors)//3 # default setting
#     anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

#     true_boxes = np.array(true_boxes)
#     input_shape = np.array(input_shape)
#     boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
#     boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
#     true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
#     true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

#     m = true_boxes.shape[0]
#     grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
#     y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
#         dtype='float32') for l in range(num_layers)]

#     # Expand dim to apply broadcasting.
#     anchors = np.expand_dims(anchors, 0)
#     anchor_maxes = anchors / 2.
#     anchor_mins = -anchor_maxes
#     valid_mask = boxes_wh[..., 0]>0

#     for b in range(m):
#         # Discard zero rows.
#         wh = boxes_wh[b, valid_mask[b]]
#         if len(wh)==0: continue
#         # Expand dim to apply broadcasting.
#         wh = np.expand_dims(wh, -2)
#         box_maxes = wh / 2.
#         box_mins = -box_maxes

#         intersect_mins = np.maximum(box_mins, anchor_mins)
#         intersect_maxes = np.minimum(box_maxes, anchor_maxes)
#         intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
#         intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
#         box_area = wh[..., 0] * wh[..., 1]
#         anchor_area = anchors[..., 0] * anchors[..., 1]
#         iou = intersect_area / (box_area + anchor_area - intersect_area)

#         # Find best anchor for each true box
#         best_anchor = np.argmax(iou, axis=-1)

#         for t, n in enumerate(best_anchor):
#             for l in range(num_layers):
#                 if n in anchor_mask[l]:
#                     i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
#                     j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
#                     k = anchor_mask[l].index(n)
#                     c = true_boxes[b,t, 4].astype('int32')
#                     y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
#                     y_true[l][b, j, i, k, 4] = 1
#                     y_true[l][b, j, i, k, 5+c] = 1
#     print(y_true)
#     return y_true              


# def get_anchors(anchors_path):
#     '''loads the anchors from a file'''
#     with open(anchors_path) as f:
#         anchors = f.readline()
#     anchors = [float(x) for x in anchors.split(',')]
#     return np.array(anchors).reshape(-1, 2)
