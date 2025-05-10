from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D,
                                     Flatten, Dense, Layer, BatchNormalization,
                                     Dropout, Reshape, TimeDistributed)
from tensorflow.keras.models import Model
from tensorflow import keras
import keras.backend as K
import tensorflow as tf
from tcn import TCN
import numpy as np


def tcn_model(input_shape):
    """
    Create a Temporal Convolutional Network (TCN) model for time series classification.

    The TCN model is composed of an Input layer,
    a TCN layer with 64 filters, a kernel size of 5, dilations = (1, 2, 4, 8, 16, 32), 
    and a Dense output layer with 10 units and a softmax activation function.

    The model is compiled using the Adam optimizer, sparse categorical crossentropy loss,
    and accuracy metric.

    Parameters
    ----------
    input_shape : tuple
        The shape of the input data, including the number of time steps and the number of features.
        For example, (100, 10) means 100 time steps and 10 features.

    Returns
    -------
    model : keras.Sequential
        A compiled Keras Sequential model with the TCN architecture.
    """
   

    model = keras.Sequential([
      TCN(input_shape=input_shape, nb_filters=64, kernel_size=5, dilations = (1, 2, 4, 8, 16, 32)),
      Dense(10, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    return model
