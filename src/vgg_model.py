from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras import Sequential

## Needed to download pre-trained weights for imagenet
import ssl
ssl._create_default_https_context = ssl._create_unverified_context



## Function to create basic VGG19 model pretrained with custom FC layers
def generate_vgg_model_basic(input_shape, classes_len):
    # Reconfigure single channel input into a greyscale 3 channel input
    img_input = Input(shape=(input_shape[0], input_shape[1], 1))
    img_conc = Concatenate()([img_input, img_input, img_input])

    # Generate VGG19 model with pre-trained imagenet weights, input as given above, without the fully connected layers
    model_base = VGG19(include_top=False, weights='imagenet', input_tensor=img_conc)

    # Add fully connected layers
    model = Sequential()
    # Start with base model consisting of convolutional layers
    model.add(model_base)

    # Flatten
    model.add(Flatten())

    # Add fully connected layers
    model.add(Dense(512, activation='relu', name='Dense_Intermediate_1'))
    model.add(Dense(32, activation='relu', name='Dense_Intermediate_2'))

    # Possible dropout for regularisation can be added later and experimented with
    # model.add(Dropout(0.1, name='Dropout_Regularization'))

    # Final output layer
    model.add(Dense(classes_len, activation='softmax', name='Output'))

    return model


## Function to generate VGG19 model adapted with extra convolutional layers due to larger image size
def generate_vgg_model_adv(input_shape, classes_len):
    # Reconfigure single channel input into a greyscale 3 channel input
    img_input = Input(shape=(input_shape[0], input_shape[1], 1))
    img_conc = Concatenate()([img_input, img_input, img_input])

    # Generate VGG19 model with pre-trained imagenet weights, input as given above, without the fully connected layers
    model_base = VGG19(include_top=False, weights='imagenet', input_tensor=img_conc)

    # On top of original VGG19 model without fully connected layers we add finer filters with further layers
    model = Sequential()
    # Start with base model consisting of convolutional layers
    model.add(model_base)

    # Generate added Convolutional layers
    model.add(Conv2D(1024, (3, 3),
                      activation='relu',
                      padding='same'))
    model.add(Conv2D(1024, (3, 3),
                      activation='relu',
                      padding='same'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    ## add fully connected layers for classification from features
    # Flatten
    model.add(Flatten())

    # Add fully connected layers
    model.add(Dense(512, activation='relu', name='Dense_Intermediate_1'))
    model.add(Dense(32, activation='relu', name='Dense_Intermediate_2'))

    # Possible dropout for regularisation can be added later and experimented with
    # model.add(Dropout(0.1, name='Dropout_Regularization'))

    ##Final output layer
    model.add(Dense(classes_len, activation='softmax', name='Output'))

    # for cnn_block_layer in model.layers[0].layers:

    return model
