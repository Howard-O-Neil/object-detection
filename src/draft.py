import numpy as np
import tensorflow as tf

a = tf.constant([1, 2, 3, 4, 5])
b = tf.constant([1, 5, 3, 4, 5])
c = tf.constant([1, 2, 9, 4, 5])

stack = tf.stack([a, b, c], axis=1)
print(stack.numpy()) 
print(stack[:, 1])


a = np.array([])
print(a[0])