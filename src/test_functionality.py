from tensorflow.python.keras import Sequential

from create_dummy_data import generate_all_data
from train_test_model import train_network, test_network
from vgg_model import generate_vgg_model_basic, generate_vgg_model_adv


##Create model
model_type = "basic"
# model_type = "advanced"

# Inputs for creating model (should be from data preprocessing
input_shape = [512,512]
# CLASSES = 3
CLASSES = 10
BATCH_SIZE = 10
EPOCH_1 = 2
EPOCH_2 = 2

# Create CNN model
if model_type == "basic":
    model = generate_vgg_model_basic(input_shape, CLASSES)
else:
    model = generate_vgg_model_adv(input_shape, CLASSES)


# Freeze VGG19 pre-trained layers
model.layers[0].trainable = False

train_x, train_y, val_x, val_y, test_x, test_y, trainAug, valAug = generate_all_data()
model = train_network(model, train_x, train_y, val_x, val_y, BATCH_SIZE, EPOCH_1, EPOCH_2)
y_pred = test_network(model, test_x)
print(y_pred)
