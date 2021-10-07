# Based on the VGGNet network introduced by Simonyan and Zisserman 
# in Very Deep Convolutional Networks for Large Scale Image Recognition.

# VGGNet-like architectures are characterized by:
# - Using only 3×3 convolutional layers stacked on top of each other 
#   in increasing depth
# - Fully-connected layers at the end of the network prior to 
#   a softmax classifier
# - Reducing volume size by max pooling

# As seen on Pyimagesearch

# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

# create SmallerVGGNet class
class SmallerVGGNet:

    # define build function 
    # receives: width, height, depth, classes and
    #   the activation function
    @staticmethod
    def build(width, height, depth, classes, finalAct="softmax"):
                
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # CONV => RELU => POOL
        # CONV layer with 32 filters with a 3x3 kernel
        model.add(Conv2D(32, (3, 3), padding="same",
            input_shape=inputShape))
        # RELU activation
        model.add(Activation("relu"))
        # apply batch normalization
        model.add(BatchNormalization(axis=chanDim))
        # apply max pooling
        model.add(MaxPooling2D(pool_size=(3, 3)))
        # apply dropout to reduce overfitting
        model.add(Dropout(0.25))

        # two sets of (CONV => RELU) * 2 => POOL
        # reduce spatial size but increasing depth

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation(finalAct))

        # return the constructed network architecture
        return model
