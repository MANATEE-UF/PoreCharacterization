from random import sample
from tkinter.tix import DirSelectDialog
import tensorflow as tf
import numpy as np
from skimage import io
import os
import time
import matplotlib.pyplot as plt
from Models.CNN_RNN import Create_CNN_RNN, CNN_RNN_Model
from Models.RNN_OneStep import Create_RNN_OneStep
from Models.ConvLSTM_OneStep import CreateConvLSTM_OneStep
from Models.ConvLSTM import CreateConvLSTM
from DataUtils.GenerateDataSet import GenerateDataSet
from DataUtils.ImageVisualization import *


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # Suppresses warning messages from tf

    dataPath = "/Users/mitchellmika/Desktop/PNG"
    numTimeSteps = 10
    stepSpacing = 3
    augmentOptions = {} # {"CROP":0, "ROTATION":True, "FLIP":True}
    imageHeight, imageWidth = GetImageSize(dataPath)

    x_train, y_train, x_val, y_val =  GenerateDataSet(
                                            dataPath, 
                                            numTimeSteps=numTimeSteps, 
                                            stepSpacing=stepSpacing, 
                                            flatten=False,
                                            predictSeq=False, 
                                            dataAugments=augmentOptions,
                                            validationRatio=0.1
                                        )

    model = CreateConvLSTM_OneStep(numTimeSteps, imageHeight, imageWidth)

    TrainModel(model, x_train, y_train, x_val, y_val, numTimeSteps, stepSpacing)

    #model.load_weights("ML_Prediction/Saved_Models/ConvLSTM_OneStep/train3_step1/SavedWeights")
    
    ShowCumulativeOneStepPredictions(model, x_val, y_val, numTimeSteps, imageHeight, imageWidth, 7)

def GetImageSize(imagesDir):
    images = os.listdir(imagesDir)

    # Get image dimensions before processing further
    sampleImage = io.imread(imagesDir + "/" + images[0])
    imageHeight = len(sampleImage)
    imageWidth = len(sampleImage[0])

    return imageHeight, imageWidth

def TrainModel(model, x_train, y_train, x_val, y_val, timeSteps, stepSize):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(), 
        loss=tf.keras.losses.binary_crossentropy # Categorical ce and softmax both produce output of all 1
        )
    
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, min_delta=0.001)
 
    history = model.fit(
        x_train, 
        y_train, 
        epochs=2,
        batch_size=20,
        validation_data=(x_val, y_val),
        callbacks=[earlyStopping]
        ) 

    dir = f"ML_Prediction/Saved_Models/{model.name}/train{timeSteps}_step{stepSize}"
    versionNum = 2
    while os.path.exists(dir):
        dir = f"ML_Prediction/Saved_Models/{model.name}/train{timeSteps}_step{stepSize}_{versionNum}"
        versionNum += 1
    
    os.mkdir(dir)
    model.save_weights(dir+"/SavedWeights")

    with open(dir+"/TrainingHistory.txt", "w") as f:
        f.write(f"Model training history: \n {history.history}")
        f.write(f"\n Training epochs: { len(history.history['loss']) }")
        f.write(f"\n Model created at {time.asctime( time.localtime(time.time()) )}")

if __name__ == "__main__":
    main()