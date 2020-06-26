import os

import cv2


def main() -> None:
    """
    Converts all PGM images to PNG format for the mini-MIAS dataset.
    :return: None
    """
    for img_pgm in os.listdir("../data/mini-MIAS/images"):
        img = cv2.imread("../data/mini-MIAS/images/{}".format(img_pgm))
        cv2.imwrite("../data/mini-MIAS/images_converted/{}.png".format(img_pgm.split(".")[0]), img)
        print("Converted {} from PGM to PNG.".format(img_pgm))
    print("Finished converting dataset.")


if __name__ == '__main__':
    main()
