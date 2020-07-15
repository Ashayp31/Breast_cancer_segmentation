import os

import pandas as pd
from pathlib import Path
import pydicom
from skimage.measure import label, regionprops


def main() -> None:
    """
    Initial dataset pre-processing for the CBIS-DDSM dataset:
        * Retrieves the path of all images and filters out the cropped images.
        * Imports original CSV files with the full image information (patient_id, left or right breast, pathology,
        image file path, etc.).
        * Filters out the cases with more than one pathology in the original csv files.
        * Merges image path which extracted on GPU machine and image pathology which is in the original csv file on image id
        * Outputs 4 CSV files.

        Generate CSV file columns:
          img: image id (e.g Calc-Test_P_00038_LEFT_CC => <case type>_<patient id>_<left or right breast>_<CC or MLO>)
          img_path: image path on the GPU machine
          label: image pathology (BENIGN or MALIGNANT)
    :return: None
    """
    csv_path = "../data/CBIS-DDSM-mask/shortened_mask_training.csv"
    as_df = pd.read_csv(csv_path)
    
    f = open("../data/CBIS-DDSM-mask/bbox_groud_truth.txt", "a")
    for i in range(as_df.shape[0]):
        string = img_path
        minr, minc, maxr, maxc = get_bbox_of_mask(mask_img_path)
        string += "," + str(minr)
        string += "," + str(minr)
        string += "," + str(minr)
        string += "," + str(minr)
        string += ",0"
        f.write(string)

    f.close()


def get_bbox_of_mask(mask_path):
    
    dataset = pydicom.dcmread(mask_path)
    array = dataset.pixel_array

    regions = regionprops(array)
    region = regions[0]
    minr, minc, maxr, maxc = region.bbox
    return minr, minc, maxr, maxc


if __name__ == '__main__':
    main()

# ----------------------------------

# Calc-Test
# data_cnt: 282
# multi pathology case:
# ['Calc-Test_P_00353_LEFT_CC', 'Calc-Test_P_00353_LEFT_MLO']

# Calc-Training
# data_cnt: 1220
# multi pathology case:
# ['Calc-Training_P_00600_LEFT_CC', 'Calc-Training_P_00600_LEFT_MLO', 'Calc-Training_P_00937_RIGHT_CC', 'Calc-Training_P_00937_RIGHT_MLO', 'Calc-Training_P_01284_RIGHT_MLO', 'Calc-Training_P_01819_LEFT_CC', 'Calc-Training_P_01819_LEFT_MLO']

# Mass-Test
# data_cnt: 359
# multi pathology case:
# ['Mass-Test_P_00969_LEFT_CC', 'Mass-Test_P_00969_LEFT_MLO']

# Mass-Training
# data_cnt: 1225
# multi pathology case:
# ['Mass-Training_P_00419_LEFT_CC', 'Mass-Training_P_00419_LEFT_MLO', 'Mass-Training_P_00797_LEFT_CC', 'Mass-Training_P_00797_LEFT_MLO', 'Mass-Training_P_01103_RIGHT_CC', 'Mass-Training_P_01103_RIGHT_MLO']
