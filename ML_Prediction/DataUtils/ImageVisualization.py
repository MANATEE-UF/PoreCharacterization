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
# Shows model prediction for one time-step in advance when data is NOT flattened
#
# Parameters:
#   model:          (tf.model)  Trained tensorflow model
#   x:              (tf.tensor) Data set that model has not seen
#   y_true          (tf.tensor) True values for x data set
#   numTimeSteps    (int)       Number of time steps that model was trained on
#   imageHeight     (int)       Height of images in data set
#   imageWidth      (int)       Width of images in data set
#
# Returns:
#   None
#########################################################################################################
def ShowOneStepPrediction(model, x, y_true, numTimeSteps, imageHeight, imageWidth):
    fig, axs = plt.subplots(5, 3, sharex="col")
    for row in axs:
        for col in row:
            col.get_xaxis().set_visible(False)
            col.get_yaxis().set_visible(False)

    axs[0][0].set_title("t")
    axs[0][1].set_title("t+1 (true)")
    axs[0][2].set_title("t+1 (pred)")

    for i in range(len(axs)):
        y_pred = tf.reshape(model.predict(tf.reshape(x[i], (1, numTimeSteps, imageHeight, imageWidth))), (imageHeight, imageWidth, 1))

        axs[i][0].imshow(x[i][-1], cmap="gray")
        axs[i][1].imshow(y_true[i], cmap="gray")
        axs[i][2].imshow(y_pred, cmap="gray")

    plt.show()

#########################################################################################################
# ShowOneStepPredictionFlat
#
# Shows model prediction for one time-step in advance when model uses flattened data
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
def ShowOneStepPredictionFlat(model, x, y_true, numTimeSteps, imageHeight, imageWidth):
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
# ShowCumulativeOneStepPredictions
#
# Shows RNN model prediction for one time-step in advance when data is NOT flattened
#
# Parameters:
#   model:          (tf.model)  Trained tensorflow model
#   x:              (tf.tensor) Data set that model has not seen
#   y_true          (tf.tensor) True values for x data set
#   numTimeSteps    (int)       Number of time steps that model was trained on
#   imageHeight     (int)       Height of images in data set
#   imageWidth      (int)       Width of images in data set
#   numPredictionSteps (int)    How many cumulative predictions should be made for visualization
#
# Returns:
#   None
#########################################################################################################
def ShowCumulativeOneStepPredictions(model, x, y_true, numTimeSteps, imageHeight, imageWidth, numPredictionSteps):
    numRows = 3 if len(x) > 3 else len(x)
    numCols = numPredictionSteps+2

    fig, axs = plt.subplots(numRows, numCols, sharex="col")
    for row in axs:
        for col in row:
            col.get_xaxis().set_visible(False)
            col.get_yaxis().set_visible(False)

    axs[0][0].set_title("t")
    axs[0][1].set_title("t+1 (true)")
    for i in range(numPredictionSteps):
        axs[0][i+2].set_title(f"t+{i+1} (pred)")


    for i in range(len(axs)):
        x_in = x[i]
        axs[i][0].imshow(x_in[-1], cmap="gray")
        axs[i][1].imshow(y_true[i], cmap="gray")

        for j in range(numPredictionSteps):
            # reshape x_in and predict based on it
            reshaped_x = tf.reshape(x_in, (1, numTimeSteps, imageHeight, imageWidth))
            y_pred = tf.reshape(model.predict(reshaped_x), (imageHeight, imageWidth, 1))

            # Use some math to turn y_pred into binary
            threshold = tf.reduce_mean(y_pred) * 1.5
            mask = y_pred > threshold
            y_pred_binary = tf.where(mask, 1.0, 0.0)
            #y_pred_binary = y_pred               # Comment this if you wanna add binarization step

            axs[i][j+2].imshow(y_pred_binary, cmap="gray")

            # Add last prediction to x_in and chop off the first prediction
            x_in = tf.concat([x_in, tf.reshape(y_pred_binary, (1,imageHeight,imageWidth,1))], axis=0)[1:numTimeSteps+1]

    plt.show()

#########################################################################################################
# ShowMultiStepPrediction
#
# Shows model prediction for architecture that predicts multiple steps
#
# Parameters:
#   model:          (tf.model)  Trained tensorflow model
#   x:              (tf.tensor) Time series to have prediction performed on
#   y:              (tf.tensor) Time series containing true values
#   numTimeSteps    (int)       Number of time steps that model was trained on
#   imageHeight     (int)       Height of images in data set
#   imageWidth      (int)       Width of images in data set
#
# Returns:
#   None
#########################################################################################################
def ShowMultiStepPrediction(model, x, y_true, numTimeSteps, imageHeight, imageWidth):

    y_pred = model.predict(tf.reshape(x, (1, numTimeSteps, imageHeight, imageWidth)))
    y_pred = tf.reshape(y_pred, (numTimeSteps, imageHeight, imageWidth, 1)) #TODO: Check if this is reversing the list order

    fig, axs = plt.subplots(2,numTimeSteps+1, sharex="col", sharey="row")

    for row in axs:
        for col in row:
            col.get_xaxis().set_visible(False)
            col.get_yaxis().set_visible(False)

    axs[0][0].set_title(f"Slice {numTimeSteps}")
    axs[0][0].imshow(tf.reshape(x[-1], (imageHeight, imageWidth,1)).numpy(), cmap="gray")
    axs[1][0].set_title(f"Slice {numTimeSteps}")
    axs[1][0].imshow(tf.reshape(x[-1], (imageHeight, imageWidth,1)).numpy(), cmap="gray")

    for i in range(1, numTimeSteps+1):
        axs[0][i].set_title(f"Slice {numTimeSteps + i}")
        axs[0][i].imshow(y_true[i-1].numpy(), cmap="gray")
        axs[1][i].imshow(y_pred[i-1].numpy(), cmap="gray")
    
    plt.show()