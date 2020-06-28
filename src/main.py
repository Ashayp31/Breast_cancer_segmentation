import argparse

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import config
from data_preprocessing import import_dataset, train_test_split_dataset


def main() -> None:
    """
    Program entry point. Parses command line arguments to decide which dataset and model to use.
    :return: None.
    """
    parse_command_line_arguments()

    # Import dataset.
    images, labels = import_dataset(data_dir="../data/{}/images_processed".format(config.dataset))

    # Split dataset.
    train_X, test_X, train_Y, test_Y = train_test_split_dataset(images, labels)

    # Construct the training image generator for data augmentation.
    augmentation = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")


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
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Verbose mode: include this flag additional print statements for debugging purposes."
                        )
    args = parser.parse_args()
    config.dataset = args.dataset
    config.verbose_mode = args.verbose


if __name__ == '__main__':
    main()
