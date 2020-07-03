import argparse
import time

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import config
from data_operations.data_generator import DataGenerator
from data_operations.data_preprocessing import encode_labels, import_dataset, dataset_stratified_split, generate_image_transforms
from model.output import evaluate
from model.train_test_model import make_predictions, train_network
from model.vgg_model import generate_vgg_model
from utils import print_runtime


def main() -> None:
    """
    Program entry point. Parses command line arguments to decide which dataset and model to use.
    :return: None.
    """
    parse_command_line_arguments()

    # Start recording time.
    start_time = time.time()

    # Create label encoder.
    l_e = LabelEncoder()

    if config.dataset == "mini-MIAS":
        # Import entire dataset.
        images, labels = import_dataset(data_dir="../data/{}/images_processed".format(config.dataset),
                                        label_encoder=l_e)

        # Split dataset into training/test/validation sets (60%/20%/20% split).
        X_train, X_test, y_train, y_test = dataset_stratified_split(split=0.20, dataset=images, labels=labels)
        X_train_rebalanced, y_train_rebalanced = generate_image_transforms(X_train, y_train)
        X_train, X_val, y_train, y_val = dataset_stratified_split(split=0.25, dataset=X_train_rebalanced,
                                                                  labels=y_train_rebalanced)

        # Create and train CNN model.
        model = generate_vgg_model(l_e.classes_.size)
        model = train_network(model, X_train, y_train, X_val, y_val, config.BATCH_SIZE, config.EPOCH_1, config.EPOCH_2)

    elif config.dataset == "CBIS-DDSM":
        df = pd.read_csv("../data/CBIS-DDSM/training-Copy1.csv")
        list_IDs = df['img_path'].values
        labels = df['label'].values

        labels = encode_labels(labels, l_e)

        X_train, X_val, y_train, y_val = dataset_stratified_split(split=0.25, dataset=list_IDs, labels=labels)

        training_generator = DataGenerator(X_train, y_train)
        validation_generator = DataGenerator(X_val, y_val)

        # Create and train CNN model.
        model = generate_vgg_model(l_e.classes_.size)
        model = train_network(model, training_generator, None, validation_generator, None, config.BATCH_SIZE, config.EPOCH_1, config.EPOCH_2)


    # Evaluate model.
    y_pred = make_predictions(model, X_val)
    if config.dataset == "mini-MIAS":
        evaluate(y_val, y_pred, l_e, config.dataset, 'N-B-M')
    elif config.dataset == "CBIS-DDSM":
        evaluate(y_val, y_pred, l_e, config.dataset, 'B-M')

    # Print training runtime.
    print_runtime("Total", round(time.time() - start_time, 2))


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


if __name__ == '__main__':
    main()
