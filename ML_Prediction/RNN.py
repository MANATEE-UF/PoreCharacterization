from multiprocessing.resource_sharer import stop
from random import sample
import tensorflow as tf
import numpy as np
from skimage import io
import os

# Generates training data set based on images found in a folder
#
# Parameters:
#   imagesDir:      (string)    Absolute path to directory where images are stored
#   numTimeSteps:   (int)       How many time steps (slices) are in a batch
#   stepSpacing:    (int)       How much spacing between steps from original data set
#   dataAugments:   (dict)      Various data augmentations that can be performed
#                   Options include:
#                       CROP:       (int)       Crops image into n number of random sections
#                       ROTATION:   (int)       Indicates how many additional rotated image data sets should be generated
#                       FLIP:       (boolean)   Indicates if flipped images should be generated
# Returns:
#   dataSet:        (tf.tensor) Data split into batches of time steps
def GenerateDataSet(imagesDir, numTimeSteps, stepSpacing = 1, dataAugments={}):
    images = os.listdir(imagesDir)
    numImages = len(images)

    # Get image dimensions before processing further
    sampleImage = io.imread(imagesDir + "/" + images[0])
    imageHeight = len(sampleImage)
    imageWidth = len(sampleImage[0])

    # # Rename files created
    # # TEMPORARY: Need to implement convention for data names
    # for i in range(numOrigImages):
    #     oldFullPath = imagesDir + "/" + origImages[i]
    #     filename = int(os.path.splitext(origImages[i])[0])
    #     filename = f"{filename:04d}.jpg"
    #     newFullPath = imagesDir + "/" + filename
    #     os.rename(oldFullPath, newFullPath)

    # Sort images in slice order
    images.sort()

    # Turn image paths into tensor objects
    for i in range(numImages):
        imagePath = imagesDir + "/" + images[i]
        readImage = tf.io.read_file(imagePath)
        images[i] = tf.io.decode_jpeg(readImage)

    # Trim original data set if it doesn't work with numTimeSteps and stepSpacing
    if numImages % numTimeSteps != 0 and numImages % stepSpacing != 0:
        multiple = stepSpacing * numTimeSteps
        rmndr = numImages % multiple
        images = images[0:numImages-rmndr]
        numImages = len(images)
    
    numOrigBatches = int(numImages / (numTimeSteps * stepSpacing))

    # Segment original images into batches
    dataSet = []
    for i in range(numOrigBatches):
        for j in range(stepSpacing):
            startIndex = i * (numTimeSteps * stepSpacing) + j
            stopIndex = (i+1) * (numTimeSteps * stepSpacing)
            dataSet.append(images[startIndex:stopIndex:stepSpacing])

    # Crop image into quadrants if requested
    # Currently crops and resizes to original size, may want to change
    # if dataAugments.has_key("CROP"):
    #     crop_height = imageHeight
    #     for batch in dataSet:
    #         newBatch = 
            

    
def main():
    trainingData = []
    testingData = []
    
    # Populate testing and training data from files on desktop
    # Training data will have three sets of sequences to train model with
    for i in range(1,5):
        stackFolder = f"/Users/mitchellmika/Desktop/Training{i}"
        images = os.listdir(stackFolder)
        images.sort()
        tempImages = []
        for j in range(220):
                image = tf.io.read_file(f"stackFolder/{j}.jpg")
                image = tf.io.decode_jpeg(image)
                image = tf.image.resize(image, [175,215])
                tempImages.append(image)
        if i < 4:
            for k in range(0,220,20):
                trainingData.append(tempImages[k:k+10])
        else:
            for k in range(0,220,20):
                testingData.append(tempImages[k:k+10])

    numSteps = 220 # number of values in time series
    training1 = trainingData[0]
    training2 = trainingData[1]
    training3 = trainingData[2]


    model = tf.keras.models.Sequential([
        tf.keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        tf.keras.layers.SimpleRNN(20, return_sequences=True),
        tf.keras.layers.Dense(10)
    ])
if __name__ == "__main__":
    #main()
    GenerateDataSet("/Users/mitchellmika/Desktop/SEM_DATA", 10, 2)
