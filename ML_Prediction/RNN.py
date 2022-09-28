from multiprocessing.resource_sharer import stop
from random import sample
import tensorflow as tf
import numpy as np
from skimage import io
import os
import matplotlib.pyplot as plt
import math

# Generates training data set based on images found in a folder
#
# Parameters:
#   imagesDir:      (string)    Absolute path to directory where images are stored
#   numTimeSteps:   (int)       How many time steps (slices) are in a batch
#   stepSpacing:    (int)       How much spacing between steps from original data set
#   dataAugments:   (dict)      Various data augmentations that can be performed
#                   Options include:
#                       CROP:       (int)       Crops image into n number of random sections
#                       ROTATION:   (boolean)   Creates additional rotated image sets
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
            dataSet.append(tf.convert_to_tensor(images[startIndex:stopIndex:stepSpacing]))

    # Crop image into quadrants if requested
    # Crops and resizes to original size
    # Multiplies data set size by number of crops requested
    # TODO: Does resizing affect porosity distribution in predictive model??? TBD
    if "CROP" in dataAugments:
        print(f"Augmenting {len(dataSet)} batches with CROP")

        # Set minimum crop height size to ensure cropped sections are not too small and lose overall detail
        # Width set based on ratio of original image
        widthHeightRatio = imageWidth / imageHeight
        minCropHeight = int(imageHeight/4)
        
        for i in range(len(dataSet)):
            batch = dataSet[i]
            print(f"Applying CROP to batch #{i}")
            for j in range(dataAugments["CROP"]):
                # Set random seed to apply for entire batch
                seed = (np.random.randint(1,dataAugments["CROP"]), np.random.randint(dataAugments["CROP"], dataAugments["CROP"]*10))

                # Set random crop size to apply for entire batch
                cropHeight = np.random.randint(minCropHeight, imageHeight)
                cropWidth = int(cropHeight * widthHeightRatio)
                cropSize = (cropHeight, cropWidth, 1)

                newBatch = []

                # Apply crop to all images in batch
                for image in batch:
                    newImage = tf.image.stateless_random_crop(image, cropSize, seed=seed)
                    newImage = tf.image.resize_with_pad(newImage, imageHeight, imageWidth)
                    newBatch.append(newImage)
                
                newBatch = tf.convert_to_tensor(newBatch)
                dataSet.append(newBatch)
    
    # Augment with 90 degree rotations
    # Images rotated 90, 180, and 270 degrees
    # Multiplies data set size by 4
    if "ROTATION" in dataAugments and dataAugments["ROTATION"]:
        print(f"Augmenting {len(dataSet)} batches with ROTATION")

        for i in range(len(dataSet)):
            print(f"Applying ROTATION to batch #{i}")
            batch = dataSet[i]

            for j in range(1,4):
                newBatch = tf.image.rot90(batch, j)
                newBatch = tf.image.resize_with_pad(newBatch, imageHeight, imageWidth)
                dataSet.append(newBatch)
    
    # Augment with vertical and horizontal flips
    # Multiplies data set size by 3
    if "FLIP" in dataAugments and dataAugments["FLIP"]:
        print(f"Augmenting {len(dataSet)} batches with FLIP")

        for i in range(len(dataSet)):
            print(f"Applying FLIP to batch #{i}")
            batch = dataSet[i]

            flipLeftRight = tf.image.flip_left_right(batch)
            flipUpDown = tf.image.flip_up_down(batch)

            dataSet.append(flipLeftRight)
            dataSet.append(flipUpDown)

    return dataSet

# Displays first image of each batch
# Dataset must be comprised of images
def PreviewDataSet(dataSet, viewTime):
    for i in range(len(dataSet)):
        img = dataSet[i][0]
        img = img.numpy()
        plt.imshow(img, cmap="gray")
        plt.title(f"Batch {i}/{len(dataSet)}")
        plt.show(block=False)
        plt.pause(viewTime)
        plt.close("all")

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
    dataSet = GenerateDataSet("/Users/mitchellmika/Desktop/SEM_DATA", 10, 2, {"CROP":2,"ROTATION":True, "FLIP":True})
    PreviewDataSet(dataSet, 0.3)
