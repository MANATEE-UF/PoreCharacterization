import tensorflow as tf

def Create_RNN_OneStep(imageSize):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.LSTM(imageSize, activation="sigmoid", return_sequences=False, input_shape=(None,imageSize)))

    model.add(tf.keras.layers.Dense(imageSize,activation="sigmoid"))

    return model
