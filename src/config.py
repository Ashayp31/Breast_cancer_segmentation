"""
Variables set by the command line arguments dictating which parts of the program to execute.
"""

# Constants
RANDOM_SEED = 111
VGG_IMG_SIZE = {
    "HEIGHT": 512,
    "WIDTH": 512   
#     "HEIGHT": 1280,
#     "WIDTH": 864   
#     "HEIGHT": 512,
#     "WIDTH": 344   
}
VGG_IMG_SIZE_LARGE = {
#     "HEIGHT": 2048,
#     "WIDTH": 2048
    "HEIGHT": 2048,
    "WIDTH": 1376
}
BATCH_SIZE = 8
EPOCH_1 = 70
EPOCH_2 = 60

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
patches = "full"
reg = "N"