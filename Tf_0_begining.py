import tensorflow as tf
import numpy as np


constant1 = tf.constant([1, 2, 3])
constant2 = tf.constant([[4, 5, 6], [7, 8, 9]])

reduce_sum = tf.reduce_sum(constant1 * constant2)

with tf.Session() as sess:
    # This will give you information in tensorflow instead of value
    print(constant1)

    print("constant1: \n{}\n".format(sess.run(constant1)))
    print("constant2: \n{}\n".format(sess.run(constant2)))
    print("constant1 * constant2: \n{}\n".format(sess.run(constant1 * constant2)))
    print("reduce_sum: \n{}\n".format(sess.run(reduce_sum)))
