"""
Variables set by the command line arguments dictating which parts of the program to execute.
"""

# Constants
RANDOM_SEED = 111
VGG_IMG_SIZE = {
    "HEIGHT": 512,
    "WIDTH": 512
}
BATCH_SIZE = 5
EPOCH_1 = 1
EPOCH_2 = 1

# Variables set by command line arguments/flags
dataset = "mini-MIAS"   # The dataset to use.
model = "basic"         # The model to use.
verbose_mode = False    # Boolean used to print additional logs for debugging purposes.
