import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

class Kaming_he_dense(keras.layers.Layer):
    fan_in = 2 # default, make no change to variables
    units = 1 # draft value
    _lambda = 0.

    def __init__(self, units, _lambda):
        super(Kaming_he_dense, self).__init__()
        self.units = units
        self._lambda = _lambda

    def build(self, input_shape):
        self.fan_in = input_shape[-1]

        he_scale = tf.math.sqrt(tf.math.divide(2., self.fan_in))
        w_init = tf.multiply(
            tf.random.normal([input_shape[-1], self.units]), 
            he_scale
        )
        b_init = tf.multiply(
            tf.random.normal([self.units]), 
            he_scale
        )

        self.w = tf.Variable(w_init, dtype=tf.float32, trainable=True)
        self.b = tf.Variable(b_init, dtype=tf.float32, trainable=True)

    def call(self, inputs):
        regularized_loss = tf.reduce_sum(
            tf.square(tf.nn.bias_add(self.w, self.b))
        )
        print(f"Layer loss = {regularized_loss}")
        self.add_loss(regularized_loss)
        return tf.nn.bias_add(tf.matmul(inputs, self.w), self.b)

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(None,5), batch_size=16),
    Kaming_he_dense(32, 0.5),
    Kaming_he_dense(32, 0.5),
    Kaming_he_dense(32, 0.5),
    Kaming_he_dense(32, 0.5),
    Kaming_he_dense(32, 0.5),
])

# test if losses will be accumulated
# forward pass multiple time
for i in range(10):
    print("===== A forward pass")
    model(np.random.normal(3, 2.5, size=(100, 5)))
    print(model.losses)

    print(len(model.losses))

# This property not longer working for custom layer
model.layers[0].trainable = False
print(model.layers[0].weights[0].trainable)