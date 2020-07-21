"""
Variables set by the command line arguments dictating which parts of the program to execute.
"""

# Constants
RANDOM_SEED = 111
VGG_IMG_SIZE = {
    "HEIGHT": 608,
    "WIDTH": 416
#     "HEIGHT": 1024,
#     "WIDTH": 640
}
VGG_IMG_SIZE_LARGE = {
    "HEIGHT": 3040,
    "WIDTH": 2080
}
BATCH_SIZE = 2
EPOCH_1 = 80
EPOCH_2 = 70

# Variables set by command line arguments/flags
dataset = "mini-MIAS"   # The dataset to use.
model = "basic"         # The model to use.
run_mode = "training"   # The type of running mode, either training or testing.
verbose_mode = False    # Boolean used to print additional logs for debugging purposes.
imagesize = "small"
segmodel = "RS50"
prep = "N"
pretrained = "imagenet"
dropout = "N"
patches = "N"