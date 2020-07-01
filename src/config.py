"""
Variables set by the command line arguments dictating which parts of the program to execute.
"""

# Constants
RANDOM_SEED = 111
VGG_IMG_HEIGHT = 512
VGG_IMG_WIDTH = 512
BATCH_SIZE = 10
EPOCH_1 = 2
EPOCH_2 = 2

# Variables set by command line arguments/flags
dataset = "mini-MIAS"   # The dataset to use.
model = "basic"         # The model to use.
verbose_mode = False    # Boolean used to print additional logs for debugging purposes.
