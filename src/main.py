from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.data_preprocessing import import_dataset, train_test_split_dataset


def main() -> None:
    """
    Program entry point.
    :return: None.
    """
    # Import dataset.
    dataset, labels = import_dataset(data_dir="../data/mini-MIAS/images_processed")

    # Split dataset.
    train_X, test_X, train_Y, test_Y = train_test_split_dataset(dataset, labels)

    # Construct the training image generator for data augmentation.
    augmentation = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")


if __name__ == '__main__':
    main()
