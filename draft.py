import tensorflow_core as tf

tf.enable_eager_execution()
# test local only
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# a = tf.constant([
#     [1, 2, 3],
#     [4, 5, 6]
# ])

# b = tf.reduce_sum(a, 1)

# print(tf.divide(a, tf.expand_dims(b, 1)).numpy())

a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

print(tf.nn.softmax(a).numpy())

