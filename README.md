# Breast-Cancer-Detection-And-Segmentation-Code

```
Thesis supervisor(s)
Dr David Harris-Birtill; Lewis McMillan

This is the first public release of the code written for the "Breast Cancer Detection and Segmentation in Mammograms using Deep Learning Techniques" dissertation, developed from 24/06/2020 to 10/08/2020.

The project is part of the MSc Artificial Intelligence at the University of St Andrews.

The project's supervisor is Dr David Harris-Birtill and co-supervisor is Lewis McMillan.
```

## Usage on a GPU lab machine

Clone the repository:

```
cd ~/Projects
git clone https://github.com/Ashayp31/Breast_cancer_segmentation
```

Create a repository that will be used to install Tensorflow 2 with CUDA 10 for Python and activate the virtual environment for GPU usage:

```
cd libraries/tf2
tar xvzf tensorflow2-cuda-10-1-e5bd53b3b5e6.tar.gz
sh build.sh
```

Activate the virtual environment:

```
source /cs/scratch/<username>/tf2/venv/bin/activate
```

Create `output`and `save_models` directories to store the results:

```
mkdir output
mkdir saved_models
```

## Run Classification Code

`cd` into the `src` directory and run the Classification Code with:

```
python whole_image_classification.py [-h] -d DATASET -m MODEL [-r RUNMODE] [-i IMAGESIZE] [-p PREPROCESS] [-do DROPOUT] [-v] [-t PRETRAINED]
```

where:
* `-h` is a  flag for help on how to run the code.
* `DATASET` is the dataset to use. Must be either `mini-MIAS` or `CBIS-DDMS`.
* `MODEL` is the model to use. Must be either `basic` or `advanced`.
* `RUNMODE` is the mode to run in (`train` or `test`). Default value is `train`.
* `IMAGESIZE` is the image size to feed into the CNN model (`small` - 512x512px; or `large` - 2048x2048px). Default value is `small`.
* `PREPROCESS` is whether you would like images to be preprocessed via image enhancemend e.g. histogram equalisation and gaussian blur. The options are `Y` for yes or `N` for no. Default is No.
* `DROPOUT` This is only valid if you are training a model. This will create a model that has dropout layers in the fully connected portion of the network. The options are `Y` for yes or `N` for no. Default is No.
* `PRETRAINED` This is only valid if you are training a model. This will create a model that is either pretrained on ImageNet or a model with random weight initialisation. The options are `imagenet` or `N` meaning random weight initialisation. The default is pretrained on imagenet.
* `-v` is a flag controlling verbose mode, which prints additional statements for debugging purposes.

## Run Segmentation Code

`cd` into the `src` directory and run the Segmentation Code with:

```
python segmentation.py [-h] -d DATASET [-r RUNMODE] [-p PREPROCESS] [-do DROPOUT] [-t PRETRAINED] [-reg REGULARISATION] [-pa CROP_TRAINING] [-c CROP_SIZE] [-v]
```

where:
* `-h` is a  flag for help on how to run the code.
* `DATASET` is the dataset to use. Must be either `mini-MIAS` or `CBIS-DDMS`.
* `RUNMODE` is the mode to run in (`train` or `test`). Default value is `train`.
* `PREPROCESS` is whether you would like images to be preprocessed via image enhancemend e.g. histogram equalisation and gaussian blur. The options are `Y` for yes or `N` for no. Default is No.
* `DROPOUT` This is only valid if you are training a model. This will create a model that has dropout layers along the encoder and decoder portions of the network. The options are `Y` for yes or `N` for no. Default is No.
* `PRETRAINED` This is only valid if you are training a model. This will create a model that is either pretrained on ImageNet or a model with random weight initialisation. The options are `imagenet` or `N` meaning random weight initialisation. The default is pretrained on imagenet.
* `REGULARISATION` This is only valid if you are training a model. This will create a model that has L2 regularisation amongst the convolutional layers within the model. The value of the L2 multiplier is hard coded and if you wish it change it will require changing the code. The options are `Y` and `N` for with and without regularisation respectively. The default is `N`.
* `CROP_TRAINING` This is only valid if you are training a model. This will give the option to train a model using the incremental cropping regime. The options are `full` or `inc`. `full` will run training using the entire image as usual where as using `inc` will run the incremental crop training regime over 3 training cycles of increasing crop sizes. The default value is `full`
* `CROP_SIZE` This is only valid if you are training a model. The will change the starting crop image size for if you want a tighter or looser starting image crop if you choose to implement incremental crop training. The options are `l` and `s` for a looser crop or tighter crop respectively. The default is `l`. If you wish to change the sizes of the crops this will need to be adjusted in the code
* `-v` is a flag controlling verbose mode, which prints additional statements for debugging purposes.

## General Running Comments

Please note that the labelling of models created and outputs are limited by a number of input and so you pay require to change the name of trained models once saved and stored in the saved_models file. Should this be the case, when testing a model you may wish to change the code directly so that it loads this model name, either by adjusting the inputs to input the model name when running the code, or by changing the name of the loaded model within the code in the whole_image_classification.py or segmentation.py file.


## Dataset usage

### mini-MIAS dataset

* This example will use the [mini-MIAS](http://peipa.essex.ac.uk/info/mias.html) dataset. After cloning the project, travel to the `data/mini-MIAS` directory (there should be 3 files in it).

* Create `images_original` and `images_processed` directories in this directory: 

```
cd data/mini-MIAS/
mkdir images_original
mkdir images_processed
```

* Move to the `images_original` directory and download the raw un-processed images:

```
cd images_original
wget http://peipa.essex.ac.uk/pix/mias/all-mias.tar.gz
```

* Unzip the dataset then delete all non-image files:

```
tar xvzf all-mias.tar.gz
rm -rf *.txt 
rm -rf README 
```

* Move back up one level and move to the `images_processed` directory. Create 3 new directories there (`benign_cases`, `malignant_cases` and `normal_cases`):

```
cd ../images_processed
mkdir benign_cases
mkdir malignant_cases
mkdir normal_cases
```

* Now run the python script for processing the dataset and render it usable with Tensorflow and Keras:

```
python3 ../../../src/dataset_processing_scripts/mini-MIAS-initial-pre-processing.py
```

### CBIS-DDSM dataset

These datasets are very large (exceeding 160GB) and more complex than the mini-MIAS dataset to use. Downloading and pre-processing them will therefore not be covered by this README. 

Our generated CSV files to use these datasets can be found in the `/data/CBIS-DDSM` directory, but the mammograms will have to be downloaded separately. The DDSM dataset can be downloaded [here](http://www.eng.usf.edu/cvprg/Mammography/Database.html), while the CBIS-DDSM dataset can be downloaded [here](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM#5e40bd1f79d64f04b40cac57ceca9272).

To run training and testing on the CBIS-DDSM dataset as used in this project one will need to downlaod the CBIS-DDSM dataset in addition to the csv files contained on the webiste.
Once done you will need to run the code in the dataset_processing_scripts/CBIS-DDSM-csv-preparation.py script. You may need to adjust some directory locations if yours differ to what is used. Upon doing so this will create a csv file that has the paths to the CBIS-DDSM images in addition to their pathology. These created files will be split into training and testing and into calcifications and masses and so if you wish to use both another csv file named "training.csv" and "testing.csv" will need to be created that join the calcification and masses files.

For segmentation similarly to follow the protocol of this project you will need to run the dataset_processing_scripts/CBIS-DDSM-csv-preparation-masks.py script. The issue with the ground truth masks in the dataset is the naming of the masks vs the ROI images changes from ending with 1-1.dcm to 1-2.dcm. As a result you will need to run the check_masks.py file found in the src dirctory. This script will print out all the names of the images that need their names changing. You will then need to copy these names into a csv named corrections_required.csv and place this file in the data/CBIS-DDSM-mask folder. This file already exists in the folder and so you may be able to already use this. After this is done you will need to run the script dataset_processing_scripts/mask_csv_changes-corrections.py to correct these image paths. Note you may need to adjust the names of files/paths if your file names differ. Once this is done the output file of this script will be ready to use for training and testing of the segmentation model.

