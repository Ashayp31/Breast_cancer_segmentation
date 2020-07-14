import argparse
import time

from data_operations.dataset_feed import create_dataset_masks
from data_operations.data_preprocessing import import_cbisddsm_training_dataset, \
    generate_image_transforms, import_cbisddsm_segmentation_training_dataset
from data_visualisation.output import evaluate, evaluate_segmentation, visualise_examples
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from utils import create_label_encoder, print_error_message, print_num_gpus_available, print_runtime
from model.train_model_segmentation import make_predictions, train_segmentation_network
from segmentation.u_net import u_net_model
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def main() -> None:
    """
        Program entry point. Parses command line arguments to decide which dataset and model to use.
        :return: None.
        """
    parse_command_line_arguments()
    print_num_gpus_available()

    # Start recording time.
    start_time = time.time()

    images, image_masks = import_cbisddsm_segmentation_training_dataset()

    # Split training dataset into training/validation sets (75%/25% split).
    X_train, X_val, y_train, y_val = train_test_split(images,
                                                        image_masks,
                                                        test_size=0.25,
                                                        random_state=config.RANDOM_SEED,
                                                        shuffle=True)
    dataset_train = create_dataset_masks(X_train, y_train)
    dataset_val = create_dataset_masks(X_val, y_val)

    
# #     image_ex, mask_ex = parse_function_segmentation_test(X_train[0], y_train[0])
#     print(y_train[0])
#     image_bytes_mask = tf.io.read_file(y_train[0])
#     image_mask = tfio.image.decode_dicom_image(image_bytes_mask, color_dim = True,  dtype=tf.uint16)
#     image_mask = tf.image.resize_with_pad(image_mask, config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'])
#     array = np.array(image_mask)
#     print(array.shape)
#     print(array[0,:,:,0].shape)
#     array = array[0,:,:,0]
    
#     image_bytes = tf.io.read_file(X_train[0])
#     image = tfio.image.decode_dicom_image(image_bytes, color_dim = True,  dtype=tf.uint16)
#     image = tf.image.resize_with_pad(image, config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'])
#     image /= 255
#     image = image[0]
#     array_2 = np.array(image)
#     print(array_2.shape)
#     print(array_2[0,:,:,0].shape)
#     array_2 = array_2[0,:,:,0]
    
#     fig,ax = plt.subplots(2, figsize=[15,15])
#     ax[0].imshow(array_2)
#     ax[1].imshow(array)
#     plt.savefig('../output/plot_images_examples.png')
    

#     exit()
    
    
    # Run in training mode.
    if config.run_mode == "train":


        # Create and train CNN model.

        model = u_net_model(input_height = config.VGG_IMG_SIZE['HEIGHT'], input_width = config.VGG_IMG_SIZE['WIDTH'])

        model = train_segmentation_network(model, dataset_train, dataset_val, config.EPOCH_1,
                                  config.EPOCH_2)


        # Save the model
        model.save("../saved_models/segmentation_model-{}_imagesize-{}x{}.h5".format(config.segmodel, str(config.VGG_IMG_SIZE['HEIGHT']),   str(config.VGG_IMG_SIZE['WIDTH'])))

    elif config.run_mode == "test":
        model = load_model("../saved_models/segmentation_model-{}_imagesize-{}x{}.h5".format(config.segmodel, str(config.VGG_IMG_SIZE['HEIGHT']), str(config.VGG_IMG_SIZE['WIDTH'])))

    # Evaluate model results.
    y_pred = make_predictions(model, dataset_val)

    evaluate_segmentation(y_val, y_pred)
    visualise_examples(X_val, y_val, y_pred)


    # Print training runtime.
    print_runtime("Total", round(time.time() - start_time, 2))


def parse_command_line_arguments() -> None:
    """
    Parse command line arguments and save their value in config.py.
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",
                        default="CBIS-DDSM",
                        help="The dataset to use. Must be either 'mini-MIAS' or 'CBIS-DDMS'."
                        )
    parser.add_argument("-m", "--model",
                        default="basic",
                        help="The model to use. Must be either 'basic' or 'advanced'."
                        )
    parser.add_argument("-r", "--runmode",
                        default="train",
                        help="Running mode: train model from scratch and make predictions, otherwise load pre-trained "
                             "model for predictions. Must be either 'train' or 'test'."
                        )
    parser.add_argument("-i", "--imagesize",
                        default="small",
                        help="small: use resized images to 512x512, otherwise use 'large' to use 2048x2048 size image with model with extra convolutions for downsizing."
                        )
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Verbose mode: include this flag additional print statements for debugging purposes."
                        )
    parser.add_argument("-s", "--segmodel",
                        default="RS50",
                        help="Segmentation model to be used."
                        )

    args = parser.parse_args()
    config.dataset = args.dataset
    config.model = args.model
    config.run_mode = args.runmode
    config.imagesize = args.imagesize
    config.verbose_mode = args.verbose
    config.segmodel = args.segmodel
    
    
def parse_function_segmentation_test(filename_original, filename_mask):
    image_bytes = tf.io.read_file(filename_original)
    image = tfio.image.decode_dicom_image(image_bytes, color_dim = True,  dtype=tf.uint16)
    image = tf.image.resize_with_pad(image[0], config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'])
    image /= 255

    image_bytes_mask = tf.io.read_file(filename_mask)
    image_mask = tfio.image.decode_dicom_image(image_bytes_mask, color_dim = True,  dtype=tf.uint16)
    image_mask = tf.image.resize_with_pad(image_mask[0], config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'])
    image_mask /= 255
    image_mask = tf.reshape(image_mask, [-1, 1])

    return image, image_mask

def convert_paths_to_arrays_test(y, if_reshape=True):

    image_bytes_mask = tf.io.read_file(y)
    image_mask = tfio.image.decode_dicom_image(image_bytes_mask,color_dim = True,  dtype=tf.uint16)
#     as_png_mask = tf.image.encode_png(image_mask[0])
#     decoded_png_mask = tf.io.decode_png(as_png_mask, channels=1)
    image_mask = tf.image.resize_with_pad(image_mask[0], config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE['WIDTH'])
    image_mask /= 255
    if if_reshape:
        image_mask = tf.reshape(image_mask, [-1, 1])
    image_as_array = np.squeeze(image_mask.numpy())
    return image_as_array


if __name__ == '__main__':
    main()
