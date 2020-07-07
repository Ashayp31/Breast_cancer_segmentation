import argparse
import time

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import config
from data_operations.data_preprocessing import encode_labels, import_dataset, dataset_stratified_split, generate_image_transforms
from model.output import evaluate
from model.train_test_model import make_predictions, train_network
from model.vgg_model import generate_vgg_model
from utils import print_runtime
import tensorflow as tf
from skimage.transform import resize
import pydicom
import matplotlib.pyplot as plt
import tensorflow_io as tfio
import numpy as np
from data_operations.dataset_feed import create_dataset



def main() -> None:

    parse_command_line_arguments()

    path = "/cs/tmp/datasets/CBIS-DDSM/Calc-Training_P_00005_RIGHT_CC/08-07-2016-DDSM-23157/1.000000-full mammogram images-38548/1-1.dcm"
    dataset = pydicom.dcmread(path)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.savefig("../output/image_original.png")
    
    image = resize(dataset.pixel_array, [512, 512])
    imgplot = plt.imshow(image, cmap=plt.cm.bone)
    plt.savefig("../output/image_resized.png")


    


def parse_command_line_arguments() -> None:
    """
    Parse command line arguments and save their value in config.py.
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",
                        default="mini-MIAS",
                        required=True,
                        help="The dataset to use. Must be either 'mini-MIAS' or 'CBIS-DDMS'."
                        )
    parser.add_argument("-m", "--model",
                        default="basic",
                        required=True,
                        help="The model to use. Must be either 'basic' or 'advanced'."
                        )
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Verbose mode: include this flag additional print statements for debugging purposes."
                        )
    args = parser.parse_args()
    config.dataset = args.dataset
    config.model = args.model
    config.verbose_mode = args.verbose
    
    
def parse_function(filename, label):
    image_bytes = tf.io.read_file(filename)
    image = tfio.image.decode_dicom_image(image_bytes,color_dim = True, scale="auto",  dtype=tf.uint16)
    as_png = tf.image.encode_png(image[0])
    decoded_png = tf.io.decode_png(as_png, channels=1)
    image = tf.image.resize(decoded_png, [512, 512])
    image /= 255

    return image, label


if __name__ == '__main__':
    main()
