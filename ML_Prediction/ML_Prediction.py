from random import sample
import tensorflow as tf
import numpy as np
from skimage import io
import os
import matplotlib.pyplot as plt
from Models.CNN_RNN import Create_CNN_RNN, CNN_RNN_Model
from Models.RNN_OneStep import Create_RNN_OneStep
from Models.ConvLSTM import CreateConvLSTM
from Data.GenerateDataSet import GenerateDataSet
from Data.ImageVisualization import ShowRNNOneStepPrediction, PreviewDataSet, ShowConvLSTMPrediction


def main():
    dataPath = "/Users/mitchellmika/Desktop/Resized_SEM_50_Binary"
    numTimeSteps = 3
    stepSpacing = 1
    augmentOptions = {"CROP":0, "ROTATION":True, "FLIP":True}
    imageHeight, imageWidth = GetImageSize(dataPath)

    x_train, y_train, x_val, y_val =  GenerateDataSet(
                                            dataPath, 
                                            numTimeSteps=numTimeSteps, 
                                            stepSpacing=stepSpacing, 
                                            flatten=False,
                                            predictSeq=True, 
                                            dataAugments=augmentOptions,
                                            validationRatio=0.1
                                        )
    #PreviewDataSet(x_train, 0.3)

    # model = Create_RNN_OneStep(imageHeight*imageWidth)
    # model = CNN_RNN_Model(numTimeSteps)
    # model.build(input_shape=(numTimeSteps, imageHeight, imageWidth, 1))
    model = CreateConvLSTM(numTimeSteps, imageHeight, imageWidth)
    print(model.summary())
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(), 
        loss=tf.keras.losses.binary_crossentropy # Categorical ce and softmax both produce output of all 1
        )
    
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.001)

    model.fit(
        x_train, 
        y_train, 
        epochs=5,
        batch_size=10,
        validation_data=(x_val, y_val),
        callbacks=[earlyStopping]
        )
    
    #ShowRNNOneStepPrediction(model, x_train[0], y_train[0], numTimeSteps, imageHeight, imageWidth)
    ShowConvLSTMPrediction(model, x_val[0], y_val[0], numTimeSteps, imageHeight, imageWidth)

def GetImageSize(imagesDir):
    images = os.listdir(imagesDir)

    # Get image dimensions before processing further
    sampleImage = io.imread(imagesDir + "/" + images[0])
    imageHeight = len(sampleImage)
    imageWidth = len(sampleImage[0])

    return imageHeight, imageWidth

if __name__ == "__main__":
    main()
    
# JPG File sequence has weird blur effect, PNG does not have this issue (lossy issues??)