# Model specification for the ATB2 grid-cell image transcriber

import tensorflow as tf

# This file provides a subclass of tf.keras.Model that serves as a
#  transcriber for the ATB2 images. It learns to make a tensor of
#  extracted digits from a tensor of the document image.

# import this file, instantiate an instance of the transcriberModel
#  class, and then either train it and save the weights, or load
#  pre-trained weights and use it for transcription.

# Model the image with hierachical convolutions and then map to output digits
class transcriberModel(tf.keras.Model):
    def __init__(self):
        # parent constructor
        super(transcriberModel, self).__init__()
        # Initial shape (36,48,3)
        self.conv1A = tf.keras.layers.Conv2D(
            32, (3, 3), strides=(2, 2), padding="valid"
        )
        self.drop1A = tf.keras.layers.Dropout(0.3)
        self.act1A = tf.keras.layers.ELU()
        # Now (18,24,32)
        self.conv1B = tf.keras.layers.Conv2D(
            64, (3, 3), strides=(2, 2), padding="valid"
        )
        self.drop1B = tf.keras.layers.Dropout(0.3)
        self.act1B = tf.keras.layers.ELU()
        # Now (9,12,64)
        self.conv1C = tf.keras.layers.Conv2D(
            128, (3, 3), strides=(2, 2), padding="valid"
        )
        self.drop1C = tf.keras.layers.Dropout(0.3)
        self.act1C = tf.keras.layers.ELU()
        # reshape to 1d
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(100)
        self.dropd1 = tf.keras.layers.Dropout(0.3)
        self.actd1 = tf.keras.layers.ELU()
        self.d2 = tf.keras.layers.Dense(100)
        self.actd2 = tf.keras.layers.ELU()
        self.dropd2 = tf.keras.layers.Dropout(0.3)
        # map directly to output format (3 digits)
        self.map_to_op = tf.keras.layers.Dense(3 * 10,)
        # softmax to get digit probabilities at each location
        self.op_reshape = tf.keras.layers.Reshape(target_shape=(3, 10,))
        self.op_softmax = tf.keras.layers.Softmax(axis=2)

    def call(self, inputs):
        x = self.conv1A(inputs)
        x = self.drop1A(x)
        x = self.act1A(x)
        x = self.conv1B(x)
        x = self.drop1B(x)
        x = self.act1B(x)
        x = self.conv1C(x)
        x = self.drop1C(x)
        x = self.act1C(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.dropd1(x)
        x = self.actd1(x)
        x = self.d2(x)
        x = self.dropd2(x)
        x = self.actd2(x)
        x = self.map_to_op(x)
        x = self.op_reshape(x)
        x = self.op_softmax(x)
        return x
