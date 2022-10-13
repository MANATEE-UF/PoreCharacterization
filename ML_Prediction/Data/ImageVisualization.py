from random import sample
import tensorflow as tf
import numpy as np
from skimage import io
import os
import matplotlib.pyplot as plt

#########################################################################################################
# PreviewDataSet()
#
# Displays a preview (first time-step) of each series in a dataset
#
# Parameters:
#   dataSet:        (tf.tensor) tf.tensors with shape (numSeries, numTimeSteps, imageHeight, imageWidth, channels)
#   viewTime:       (float)     Value indicating how long each image will be displayed
#
# Returns:
#   None
#########################################################################################################
def PreviewDataSet(dataSet, viewTime):
    for i in range(len(dataSet)):
        img = dataSet[i][0]
        img = img.numpy()
        plt.imshow(img, cmap="gray")
        plt.title(f"Series {i}/{len(dataSet)}")
        plt.show(block=False)
        plt.pause(viewTime)
        plt.close("all")


#########################################################################################################
# ShowOneStepPrediction
#
# Shows model prediction for one time-step in advance
#
# Parameters:
#   model:          (tf.model)  Trained tensorflow model
#   x:              (tf.tensor) Time series to have prediction performed on
#   y_true          (tf.tensor) Tensor containing true image value
#   numTimeSteps    (int)       Number of time steps that model was trained on
#   imageHeight     (int)       Height of images in data set
#   imageWidth      (int)       Width of images in data set
#
# Returns:
#   None
#########################################################################################################
def ShowOneStepPrediction(model, x, y_true, numTimeSteps, imageHeight, imageWidth):
    y = model.predict(tf.reshape(x, (1, numTimeSteps, imageHeight*imageWidth)))

    fig, axs = plt.subplots(2,4)
    for row in axs:
        for col in row:
            col.get_xaxis().set_visible(False)
            col.get_yaxis().set_visible(False)

    axs[0][0].set_title("t-2")
    axs[0][0].imshow(tf.reshape(x[-3], (imageHeight, imageWidth,1)).numpy(), cmap="gray")
    axs[1][0].imshow(tf.reshape(x[-3], (imageHeight, imageWidth,1)).numpy(), cmap="gray")

    axs[0][1].set_title("t-1")
    axs[0][1].imshow(tf.reshape(x[-2], (imageHeight, imageWidth,1)).numpy(), cmap="gray")
    axs[1][1].imshow(tf.reshape(x[-2], (imageHeight, imageWidth,1)).numpy(), cmap="gray")

    axs[0][2].set_title("t")
    axs[0][2].imshow(tf.reshape(x[-1], (imageHeight, imageWidth,1)).numpy(), cmap="gray")
    axs[1][2].imshow(tf.reshape(x[-1], (imageHeight, imageWidth,1)).numpy(), cmap="gray")

    axs[0][3].set_title("t+1 (True)")
    axs[0][3].imshow(tf.reshape(y_true, (imageHeight, imageWidth,1)).numpy(), cmap="gray")

    axs[1][3].set_title("t+1 (Predicted)")
    axs[1][3].imshow(tf.reshape(y, (imageHeight, imageWidth,1)).numpy(), cmap="gray")

    plt.show()


#########################################################################################################
# ShowMultipleStepPrediction
#
# Shows model prediction for one time-step in advance
#
# Parameters:
#   model:          (tf.model)  Trained tensorflow model
#   x:              (tf.tensor) Time series to have prediction performed on
#   numTimeSteps    (int)       Number of time steps that model was trained on
#   imageHeight     (int)       Height of images in data set
#   imageWidth      (int)       Width of images in data set
#
# Returns:
#   None
#########################################################################################################
def ShowMultipleStepPrediction(model, x, numTimeSteps, imageHeight, imageWidth):
    pass