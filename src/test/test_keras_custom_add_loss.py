from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
from tensorflow.keras import Sequential

import tensorflow as tf


class MyActivityRegularizer(Layer):
    """Layer that creates an activity sparsity regularization loss."""

    def __init__(self, rate=1e-2):
        super(MyActivityRegularizer, self).__init__()
        self.rate = rate

    def call(self, inputs):
        print(inputs.shape)
        # We use `add_loss` to create a regularization loss
        # that depends on the inputs.
        self.add_loss(self.rate * tf.reduce_sum(tf.square(inputs)))
        return inputs


class SparseMLP(Layer):
    """Stack of Linear layers with a sparsity regularization loss."""

    def __init__(self, output_dim):
        super(SparseMLP, self).__init__()
        self.dense_1 = layers.Dense(32, activation=tf.nn.relu)
        self.regularization = MyActivityRegularizer(1e-2)
        self.dense_2 = layers.Dense(output_dim)

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.regularization(x)
        return self.dense_2(x)

seq_model = Sequential([
    SparseMLP(32),
    SparseMLP(32),
])

y = seq_model(tf.ones((10, 12)))

print(seq_model.summary())
print(seq_model.losses)  # List containing one float32 scalar
