import tensorflow as tf

def Create_CNN_RNN():
    model = tf.keras.Sequential()

    #TODO: Workaround for keras assuming None as first dimension
    # Conv2D only works on 3D and 4D tensors, so need a work around in order to process multiple images in single batch

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=(1,1), input_shape=(15,50,50,1)))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=(1,1)))
    model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=5, strides=(1,1)))
    model.add(tf.keras.layers.Conv2D(filters=4, kernel_size=5, strides=(1,1)))
    model.add(tf.keras.layers.Conv2D(filters=2, kernel_size=5, strides=(1,1)))
    model.add(tf.keras.layers.Conv2D(filters=1, kernel_size=5, strides=(1,1)))
    model.add(tf.keras.layers.Reshape((15,26,26,1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=1))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Reshape((15,26**2)))

    model.add(tf.keras.layers.SimpleRNN(676, return_sequences=True))
    model.add(tf.keras.layers.SimpleRNN(676, return_sequences=True))
    model.add(tf.keras.layers.SimpleRNN(676))

    model.add(tf.keras.layers.Dense(2500, activation="sigmoid"))

    return model

class CNN_RNN_Model(tf.keras.Model):
    def __init__(self, numTimeSteps):
        super(CNN_RNN_Model, self).__init__()
        self.numTimeSteps = numTimeSteps

        self.Conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=(1,1), input_shape=(numTimeSteps,50,50,1))
        self.Conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=(1,1))
        self.Conv3 = tf.keras.layers.Conv2D(filters=8, kernel_size=5, strides=(1,1))
        self.Conv4 = tf.keras.layers.Conv2D(filters=4, kernel_size=5, strides=(1,1))
        self.Conv5 = tf.keras.layers.Conv2D(filters=2, kernel_size=5, strides=(1,1))
        self.Conv6 = tf.keras.layers.Conv2D(filters=1, kernel_size=5, strides=(1,1))
        self.Pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=1)

        self.Flatten1 = tf.keras.layers.Flatten()

        self.RNN1 = tf.keras.layers.SimpleRNN(625, return_sequences=True)
        self.RNN2 = tf.keras.layers.SimpleRNN(625, return_sequences=True)
        self.RNN3 = tf.keras.layers.SimpleRNN(625)

        self.Dense1 = tf.keras.layers.Dense(2500, activation="sigmoid")

    def call(self, x):
        x = tf.reshape(x, (self.numTimeSteps, 50, 50, 1))

        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Conv5(x)
        x = self.Conv6(x)
        
        x = self.Pool1(x)

        x = self.Flatten1(x)

        x = tf.reshape(x, (1, self.numTimeSteps, 25**2))

        x = self.RNN1(x)
        x = self.RNN2(x)
        x = self.RNN3(x)

        x = self.Dense1(x)

        return x
