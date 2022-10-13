from random import sample
import tensorflow as tf
import numpy as np
from skimage import io
import os
import matplotlib.pyplot as plt
from Models.CNN_RNN import Create_CNN_RNN, CNN_RNN_Model
from Models.RNN_OneStep import Create_RNN_OneStep
from Data.GenerateDataSet import GenerateDataSet
from Data.ImageVisualization import ShowOneStepPrediction, PreviewDataSet, ShowMultipleStepPrediction


def main():
    dataPath = "/Users/mitchellmika/Desktop/Resized_SEM_50"
    numTimeSteps = 10
    stepSpacing = 3
    augmentOptions = {"CROP":0, "ROTATION":True, "FLIP":True}
    imageHeight, imageWidth = GetImageSize(dataPath)

    x_train, y_train, x_test, y_test =  GenerateDataSet(
                                            dataPath, 
                                            numTimeSteps=numTimeSteps, 
                                            stepSpacing=stepSpacing, 
                                            flatten=False,
                                            predictSeq=False, 
                                            dataAugments=augmentOptions
                                        )
    #PreviewDataSet(x_train, 0.3)

    #model = Create_RNN_OneStep(imageHeight*imageWidth)
    model = CNN_RNN_Model(numTimeSteps)
    model.build(input_shape=(numTimeSteps, imageHeight, imageWidth, 1))
    print(model.summary())
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
        loss="mse",
        metrics=["mae"]
        )
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    model.fit(
        x_train, 
        y_train, 
        epochs=30,
        batch_size=1,
        callbacks=[callback]
        )
    
    model.evaluate(x_test, y_test, batch_size=1)
    
    ShowOneStepPrediction(model, x_train[0], y_train[0], numTimeSteps, imageHeight, imageWidth)

def GetImageSize(imagesDir):
    images = os.listdir(imagesDir)

    # Get image dimensions before processing further
    sampleImage = io.imread(imagesDir + "/" + images[0])
    imageHeight = len(sampleImage)
    imageWidth = len(sampleImage[0])

    return imageHeight, imageWidth

if __name__ == "__main__":
    main()
    
