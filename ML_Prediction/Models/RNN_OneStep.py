import tensorflow as tf

def Create_RNN_OneStep(imageSize):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.SimpleRNN(1000, return_sequences=True, input_shape=(None,imageSize)))
    model.add(tf.keras.layers.SimpleRNN(1000))

    model.add(tf.keras.layers.Dense(imageSize,activation="sigmoid"))

    return model
