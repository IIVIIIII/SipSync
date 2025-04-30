from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D,
                                     Flatten, Dense, Layer, BatchNormalization,
                                     Dropout, Reshape, TimeDistributed)
from tensorflow.keras.models import Model
from tensorflow import keras
import keras.backend as K
import tensorflow as tf
from tcn import TCN
import numpy as np


def cnn_model(input_shape):
    """
    Create a Convolutional Neural Network (CNN) model for time series classification.

    The CNN architecture consists of three convolutional blocks followed by fully connected layers.
    Each convolutional block has the following layers:
    1. Conv2D layer with 20 filters, a kernel size of (3, 3), and ReLU activation
    2. MaxPooling2D layer with a pool size of (3, 1)
    3. Dropout layer with a dropout rate of 0.15

    After the convolutional blocks, the following fully connected layers are added:
    1. Flatten layer to convert the 2D feature maps into a 1D feature vector
    2. Dense layer with 64 units and ReLU activation
    3. Dense layer with 32 units and ReLU activation
    4. Dense layer with 10 units and softmax activation (output layer)

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
      A compiled Keras Sequential model with the CNN architecture.

    Example
    -------
    >>> input_shape = (100, 10)  # 100 time steps and 10 features
    >>> model = cnn_model(input_shape)
    >>> print(model.summary())

    # Prepare your dataset
    >>> x_train, y_train, x_test, y_test = ...  # Load or preprocess your data

    # Train the model
    >>> history = model.fit(x_train, y_train, epochs=10, batch_size=32,
                            validation_data=(x_test, y_test))

    # Evaluate the model
    >>> loss, accuracy = model.evaluate(x_test, y_test)
    >>> print(f"Test loss: {loss}, Test accuracy: {accuracy}")

    # Make predictions
    >>> y_pred = model.predict(x_test)
    >>> y_pred_classes = np.argmax(y_pred, axis=1)
    """
    # Note: Please use the Sequential API (i.e., keras.Sequential)
    # when building your model; do not use the Functional API.
    # YOUR CODE HERE

    model = keras.Sequential([
      Conv2D(20, kernel_size=(3, 3), activation='relu', input_shape=(input_shape)),
      MaxPooling2D(pool_size=(3, 1)),
      Dropout(0.15),
      Conv2D(20, kernel_size=(3, 3), activation='relu'),
      MaxPooling2D(pool_size=(3, 1)),
      Dropout(0.15),
      Conv2D(20, kernel_size=(3, 3), activation='relu'),
      MaxPooling2D(pool_size=(3, 1)),
      Dropout(0.15),

      Flatten(),
      Dense(64, activation='relu'),
      Dense(32, activation='relu'),
      Dense(16, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

    pass


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

    Example
    -------
    >>> input_shape = (100, 10)  # 100 time steps and 10 features
    >>> model = tcn_model(input_shape)
    >>> print(model.summary())

    # Prepare your dataset
    >>> x_train, y_train, x_test, y_test = ...  # Load or preprocess your data

    # Train the model
    >>> history = model.fit(x_train, y_train, epochs=10, batch_size=32,
                            validation_data=(x_test, y_test))

    # Evaluate the model
    >>> loss, accuracy = model.evaluate(x_test, y_test)
    >>> print(f"Test loss: {loss}, Test accuracy: {accuracy}")

    # Make predictions
    >>> y_pred = model.predict(x_test)
    >>> y_pred_classes = np.argmax(y_pred, axis=1)
    """
    # Hint: look at the syntax of the keras-tcn package
    # Note: Please use the Sequential API (i.e., keras.Sequential)
    # when building your model; do not use the Functional API.
    # YOUR CODE HERE

    



    # The TCN model is composed of an Input layer,
    # a TCN layer with 64 filters, a kernel size of 5, dilations = (1, 2, 4, 8, 16, 32), 
    # and a Dense output layer with 10 units and a softmax activation function.

    # The model is compiled using the Adam optimizer, sparse categorical crossentropy loss,
    # and accuracy metric.

    model = keras.Sequential([
      TCN(input_shape=input_shape, nb_filters=64, kernel_size=5, dilations = (1, 2, 4, 8, 16, 32)),
      # Dense(16, activation='softmax')
      Dense(10, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    return model


    pass
