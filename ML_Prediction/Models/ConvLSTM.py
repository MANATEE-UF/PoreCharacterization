from argparse import ArgumentError
import tensorflow as tf
import numpy as np

def CreateConvLSTM(numTimeSteps, imageHeight, imageWidth):
    if imageHeight != imageWidth:
        raise ArgumentError("Make the image square or I don't care")

    model = tf.keras.Sequential([
        tf.keras.layers.ConvLSTM2D(
            filters=imageHeight,
            kernel_size=(5,5),
            padding="same",
            return_sequences=True,
            activation="relu",
            input_shape=(numTimeSteps, imageHeight, imageWidth, 1)
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ConvLSTM2D(
            filters=imageHeight,
            kernel_size=(3,3),
            padding="same",
            return_sequences=True,
            activation="relu"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ConvLSTM2D(
            filters=imageHeight,
            kernel_size=(1,1),
            padding="same",
            return_sequences=True,
            activation="relu"
        ),
        tf.keras.layers.Conv3D(filters=1, kernel_size=(3,3,3), activation="sigmoid",padding="same"),
    ])

    return model
