import argparse
import time
from data_operations.dataset_feed import create_dataset_masks, create_dataset_patched, create_dataset_cropped
from data_operations.data_preprocessing import import_cbisddsm_training_dataset, \
    generate_image_transforms, import_cbisddsm_segmentation_training_dataset, import_cbisddsm_segmentation_testing_dataset
from data_visualisation.output import evaluate, evaluate_segmentation, visualise_examples
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from utils import create_label_encoder, print_error_message, print_num_gpus_available, print_runtime
from model.train_model_segmentation import *
from model.u_net_RES import u_net_res_model
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError, binary_crossentropy
from model.train_model_segmentation_upsizing import train_segmentation_network_incremental
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

def main() -> None:

    """
        Program entry point. Parses command line arguments to decide which dataset and model to use.
        :return: None.
        Script used for training and testing a segmentation model on the CBIS-DDSM dataset
        """
    parse_command_line_arguments()
    print_num_gpus_available()
    

    # Start recording time.
    start_time = time.time()

    if config.run_mode == "train":

        images, image_masks = import_cbisddsm_segmentation_training_dataset()

        if config.patches == "inc":
            train_segmentation_network_incremental(images, image_masks)
        else:
            # Split training dataset into training/validation sets (75%/25% split).
            X_train, X_val, y_train, y_val = train_test_split(images,
                                                                image_masks,
                                                                test_size=0.25,
                                                                random_state=config.RANDOM_SEED,
                                                                shuffle=True)
            if config.patches == "full":
                dataset_train = create_dataset_masks(X_train, y_train)
                dataset_val = create_dataset_masks(X_val, y_val)
            elif config.patches == "patch":
                dataset_train = create_dataset_patched(X_train, y_train)
                dataset_val = create_dataset_patched(X_val, y_val)


            # Run in training mode.
            if config.run_mode == "train":
                # Create and train CNN model.
                model = u_net_res_model(input_height = config.VGG_IMG_SIZE['HEIGHT'], input_width = config.VGG_IMG_SIZE['WIDTH'])
                print()
                print("TRAINING MODEL...")
                print()
                model = train_segmentation_network(model, dataset_train, dataset_val, config.EPOCH_1,
                                                          config.EPOCH_2)


                    # Save the model
                model.save("../saved_models/segmentation_model-{}_imagesize-{}x{}_filtered_{}.h5".format(config.segmodel, str(config.VGG_IMG_SIZE['HEIGHT']),   str(config.VGG_IMG_SIZE['WIDTH']), config.prep))
                
        # print training time      
        print_runtime("Total", round(time.time() - start_time, 2))
        
        # Evaluate model results.
        if config.patches == "full":
            y_pred = make_predictions(model, dataset_val)
            evaluate_segmentation(y_val, y_pred, threshold = 0.5)
            visualise_examples(X_val, y_val, y_pred, threshold = 0.5)
        else:
            y_pred = make_predictions(model, dataset_val)
            evaluate_segmentation(dataset_val, y_pred, threshold = 0.5)
            visualise_examples(dataset_val, None, y_pred, threshold = 0.5)

        
    elif config.run_mode == "test":
        print()
        print("TESTING MODEL...")
        print()
            
        model = load_model("../saved_models/segmentation/segmentation_model-RS50_imagesize-1024x640_filtered_Y_dual_loss_dropout.h5", compile=False)
#         model.load_weights("../saved_models/segmentation/segmentation_model-upsizing_checkpoint_0_small_start.h5")

        images, image_masks = import_cbisddsm_segmentation_testing_dataset()
        if config.patches == "inc":
            dataset_test = create_dataset_cropped(images, image_masks, config.BATCH_SIZE)
        else:
            dataset_test = create_dataset_masks(images, image_masks)
        
        training_start_time = time.time()

        if config.patches == "full":
            y_pred = make_predictions(model, dataset_test)
            print_runtime("Total testing time", round(time.time() - training_start_time, 2))
            
            evaluate_segmentation(image_masks, y_pred, threshold = 0.5)
            visualise_examples(images, image_masks, y_pred, threshold = 0.5)
        else:
            y_pred = make_predictions(model, dataset_test)
            print_runtime("Total testing time", round(time.time() - training_start_time, 2))
#             evaluate_segmentation(dataset_test, y_pred, threshold = 0.5)
            visualise_examples(dataset_test, None, y_pred, threshold = 0.5)

    


           
    # Print training runtime.


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
    parser.add_argument("-p", "--prep",
                        default="N",
                        help="Preprocessing of images"
                        )
    parser.add_argument("-t", "--pretrained",
                        default="imagenet",
                        help="pretrained weights for the model. Use none if to use no pretrained weights"
                        )
    parser.add_argument("-do", "--dropout",
                        default="N",
                        help="Whether to include dropout in the network, change to Y to include. Only in the contracting layers"
                        )
    parser.add_argument("-pa", "--patches",
                        default="full",
                        help="Whether to train a on image patches or whole image, or incremental upsizing done with full, patch, "
                        "or inc"
                        )
    parser.add_argument("-reg", "--reg",
                        default="N",
                        help="Whether to apply regularisation, deafault as N, change to Y for applying regularisation"
                        )
    args = parser.parse_args()
    config.dataset = args.dataset
    config.model = args.model
    config.run_mode = args.runmode
    config.imagesize = args.imagesize
    config.verbose_mode = args.verbose
    config.segmodel = args.segmodel
    config.prep = args.prep
    config.pretrained = args.pretrained
    config.dropout = args.dropout
    config.patches = args.patches
    config.reg = args.reg


if __name__ == '__main__':
    main()
