from random import sample
import tensorflow as tf
import numpy as np
from skimage import io
import os
import matplotlib.pyplot as plt
import time

#########################################################################################################
# GenerateDataSet()
#
# Generates training data set based on images found in a folder
#
# Parameters:
#   imagesDir:      (string)    Absolute path to directory where images are stored
#   numTimeSteps:   (int)       How many time steps (slices) are in a series
#   stepSpacing:    (int)       How much spacing between steps from original data set **
#   flatten:        (bool)      Determines if image should be flattened in data sets. Needed if no CNN is used in model **
#   predictSeq:     (bool)      Determines if y_train is a single value or sequence **
#   dataAugments:   (dict)      Various data augmentations that can be performed **
#                   Options include:
#                       CROP:       (int)       Crops image into n number of random sections
#                       ROTATION:   (boolean)   Creates additional rotated image sets
#                       FLIP:       (boolean)   Indicates if flipped images should be generated
# Returns:
#   x_train:        (tf.tensor) tf.tensor with shape (numSeries, numTimeSteps, imageHeight, imageWidth, channels)
#                               OR tf.tensor with shape (numSeries, numTimeSteps, imageHeight * imageWidth)
#   y_train:        (tf.tensor) tf.tensor with shape (numSeries, numTimeSteps, imageHeight * imageWidth)
#                               OR tf.tensor with shape (numSeries, imageHeight * imageWidth)
#   x_test:         (tf.tensor) tf.tensor with shape (numSeries, numTimeSteps, imageHeight, imageWidth, channels)
#                               OR tf.tensor with shape (numSeries, numTimeSteps, imageHeight * imageWidth)
#   y_test:         (tf.tensor) tf.tensor with shape (numSeries, numTimeSteps, imageHeight * imageWidth)
#                               OR tf.tensor with shape (numSeries, imageHeight * imageWidth)
#########################################################################################################
def GenerateDataSet(    imagesDir, 
                        numTimeSteps, 
                        stepSpacing=1, 
                        flatten=False,
                        predictSeq=False, 
                        dataAugments={}):
    
    numTimeSteps += 1   # Allows for y_train to also be generated, each training set will still only contain
                        #   original numTimeSteps number of steps
    # TODO: Check if y_train can be generated a different way to maximize data usage

    images = os.listdir(imagesDir)
    numImages = len(images)

    # Get image dimensions before processing further
    sampleImage = io.imread(imagesDir + "/" + images[0])
    imageHeight = len(sampleImage)
    imageWidth = len(sampleImage[0])

    # Sort images in slice order
    images.sort()

    # Turn image paths into tensor objects
    for i in range(numImages):
        imagePath = imagesDir + "/" + images[i]
        readImage = tf.io.read_file(imagePath)
        images[i] = tf.cast(tf.io.decode_jpeg(readImage, channels=1), dtype=tf.float32) / 255.0

    # Trim original data set if it doesn't work with numTimeSteps and stepSpacing
    if numImages % numTimeSteps != 0 and numImages % stepSpacing != 0:
        multiple = stepSpacing * numTimeSteps
        rmndr = numImages % multiple
        images = images[0:numImages-rmndr]
        numImages = len(images)
    
    numOrigSeries = int(numImages / (numTimeSteps * stepSpacing))

    # Segment original images into time series
    dataSet = []
    for i in range(numOrigSeries):
        for j in range(stepSpacing):
            startIndex = i * (numTimeSteps * stepSpacing) + j
            stopIndex = (i+1) * (numTimeSteps * stepSpacing)
            dataSet.append(tf.convert_to_tensor(images[startIndex:stopIndex:stepSpacing]))

    # Crop image into quadrants if requested
    # Crops and resizes to original size
    # Multiplies data set size by number of crops requested
    # TODO: Does resizing affect porosity distribution in predictive model??? TBD
    if "CROP" in dataAugments and dataAugments["CROP"] > 0:
        print(f"Augmenting {len(dataSet)} series with CROP")
        time.sleep(1.5)

        # Set minimum crop height size to ensure cropped sections are not too small and lose overall detail
        # Width set based on ratio of original image
        widthHeightRatio = imageWidth / imageHeight
        minCropHeight = int(imageHeight/2)
        
        for i in range(len(dataSet)):
            series = dataSet[i]
            print(f"Applying CROP to series #{i}")
            for j in range(dataAugments["CROP"]):
                # Set random seed to apply for entire series
                seed = (np.random.randint(1,dataAugments["CROP"]*10), np.random.randint(dataAugments["CROP"]*10, dataAugments["CROP"]*100))

                # Set random crop size to apply for entire series
                cropHeight = np.random.randint(minCropHeight, imageHeight)
                cropWidth = int(cropHeight * widthHeightRatio)
                cropSize = (cropHeight, cropWidth, 1)

                newSeries = []

                # Apply crop to all images in series
                for image in series:
                    newImage = tf.image.stateless_random_crop(image, cropSize, seed=seed)
                    newImage = tf.image.resize_with_pad(newImage, imageHeight, imageWidth)
                    newSeries.append(newImage)
                
                newSeries = tf.convert_to_tensor(newSeries)
                dataSet.append(newSeries)
    
    # Augment with 90 degree rotations
    # Images rotated 90, 180, and 270 degrees
    # Multiplies data set size by 4
    if "ROTATION" in dataAugments and dataAugments["ROTATION"]:
        print(f"Augmenting {len(dataSet)} series with ROTATION")
        time.sleep(1.5)

        for i in range(len(dataSet)):
            print(f"Applying ROTATION to series #{i}")
            series = dataSet[i]

            for j in range(1,4):
                newSeries = tf.image.rot90(series, j)
                newSeries = tf.image.resize_with_pad(newSeries, imageHeight, imageWidth)
                dataSet.append(newSeries)
    
    # Augment with vertical and horizontal flips
    # Multiplies data set size by 3
    if "FLIP" in dataAugments and dataAugments["FLIP"]:
        print(f"Augmenting {len(dataSet)} series with FLIP")
        time.sleep(1.5)

        for i in range(len(dataSet)):
            print(f"Applying FLIP to series #{i}")
            series = dataSet[i]

            flipLeftRight = tf.image.flip_left_right(series)
            flipUpDown = tf.image.flip_up_down(series)

            dataSet.append(flipLeftRight)
            dataSet.append(flipUpDown)

    # Reshape data set to [timestep, imageHeight * imageWidth]
    if flatten:
        for i in range(len(dataSet)):
            dataSet[i] = tf.reshape(dataSet[i], (numTimeSteps, imageHeight * imageWidth))
    
    dataSet = tf.random.shuffle(dataSet)

    # Separate dataSet into x_train and y_train
    x_train = []
    y_train = []
    for series in dataSet:
        x_train.append(series[:numTimeSteps-1])
        if predictSeq:
            y_train.append(series[1:numTimeSteps])
        else:
            y_train.append(series[-1])


    # TODO: Create x_test and y_test from training data set
    x_test = []
    y_test = []
    numSeries = len(x_train)
    numTestSeries = int(numSeries * .2)
    for i in range(numTestSeries):
        x_test.append(x_train.pop(-1))
        y_test.append(y_train.pop(-1))

    print()
    x_train = tf.convert_to_tensor(x_train)
    print(f"Shape of x_train: {tf.shape(x_train)}")

    y_train = tf.convert_to_tensor(y_train)
    print(f"Shape of y_train: {tf.shape(y_train)}")

    x_test = tf.convert_to_tensor(x_test)
    print(f"Shape of x_test:  {tf.shape(x_test)}")

    y_test = tf.convert_to_tensor(y_test)
    print(f"Shape of y_train: {tf.shape(y_test)}")

    print()
    time.sleep(3)

    return x_train, y_train, x_test, y_test
