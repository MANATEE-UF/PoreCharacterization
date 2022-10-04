from random import sample
import tensorflow as tf
import numpy as np
from skimage import io
import os
import matplotlib.pyplot as plt
import time
from Models.CNN_RNN import Create_CNN_RNN
from Models.RNN_OneStep import Create_RNN_OneStep


def main():
    dataPath = "/Users/mitchellmika/Desktop/SEM_DATA_BINARY_100"
    numTimeSteps = 20
    stepSpacing = 1
    augmentOptions = {"CROP":1, "ROTATION":True, "FLIP":True}
    imageHeight, imageWidth = CheckImageSize(dataPath)

    x_train, y_train = GenerateDataSet( dataPath, 
                                        numTimeSteps=numTimeSteps, 
                                        stepSpacing=stepSpacing, 
                                        flatten=True,
                                        predictSeq=False, 
                                        dataAugments=augmentOptions)
    # PreviewDataSet(dataSet, 0.3)

    model = Create_RNN_OneStep(imageHeight*imageWidth)
    print(model.summary())
    
    model.compile(optimizer="adam", loss="mae")
    model.fit(  x_train, 
                y_train, 
                epochs=1,
                batch_size=10)
    
    ShowPrediction(model, x_train[0], numTimeSteps, imageHeight, imageWidth)


#########################################################################################################
# GenerateDataSet()
#
# Generates training data set based on images found in a folder
#
# Parameters:
#   imagesDir:      (string)    Absolute path to directory where images are stored
#   numTimeSteps:   (int)       How many time steps (slices) are in a series
#   stepSpacing:    (int)       How much spacing between steps from original data set **
#   flatten:        (bool)      Determines if image should be flattened in data sets **
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
        images[i] = tf.round(tf.cast(tf.io.decode_jpeg(readImage), dtype=tf.float32) / 255.0) # Ensures binary image with value of 0 or 1

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
        time.sleep(1)

        # Set minimum crop height size to ensure cropped sections are not too small and lose overall detail
        # Width set based on ratio of original image
        widthHeightRatio = imageWidth / imageHeight
        minCropHeight = int(imageHeight/4)
        
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
        time.sleep(1)

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
        time.sleep(1)

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
    x_train = tf.convert_to_tensor(x_train)
    print("Shape of x_train:")
    print(tf.shape(x_train))
    y_train = tf.convert_to_tensor(y_train)
    print()
    print("Shape of y_train:")
    print(tf.shape(y_train))

    return x_train, y_train

# FIXME: Might not work with dataSet being moved to tf.tensor data type
#########################################################################################################
# PreviewDataSet(dataSet, viewTime)
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

def CheckImageSize(imagesDir):
    images = os.listdir(imagesDir)

    # Get image dimensions before processing further
    sampleImage = io.imread(imagesDir + "/" + images[0])
    imageHeight = len(sampleImage)
    imageWidth = len(sampleImage[0])

    return imageHeight, imageWidth

def ShowPrediction(model, x, numTimeSteps, imageHeight, imageWidth):
    y = model.predict(tf.reshape(x, (1, numTimeSteps, imageHeight*imageWidth)))
    fig, axs = plt.subplots(1,4)
    axs[0].set_title("t-3")
    axs[0].imshow(tf.reshape(x[-3], (imageHeight, imageWidth,1)).numpy(), cmap="gray")

    axs[1].set_title("t-2")
    axs[1].imshow(tf.reshape(x[-2], (imageHeight, imageWidth,1)).numpy(), cmap="gray")

    axs[2].set_title("t-1")
    axs[2].imshow(tf.reshape(x[-1], (imageHeight, imageWidth,1)).numpy(), cmap="gray")

    axs[3].set_title("t+1")
    axs[3].imshow(tf.reshape(y, (imageHeight, imageWidth,1)).numpy(), cmap="gray")

    plt.show()

if __name__ == "__main__":
    main()
    
