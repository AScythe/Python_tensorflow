import tensorflow as tf
import numpy as np

# tensorflow uses float32
x = np.random.rand(100).astype(np.float32)
# the target funcdtion
y = x*2.5 + 7


# this is used to give a range for AI because it's a test
# tf.Variable(tf.random_normal([shape], range))
Weights = tf.Variable(tf.random_normal([1], -3., 3.))

# tensorflow suggest us the value of biases should not be zero at the beginning
biases = tf.Variable(tf.zeros([1]) + 0.01)
y_pre = x*Weights + biases

# normally we use root-mean-square as the loss in linear regression case
loss = tf.reduce_mean(tf.square(y_pre-y))
# Gardient Descent is the method we use to minimize the loss. 0.5 is the learning rate (kind of the distance of each step)
# you can also combine optimizer and train together as "train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)"
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# With the exception of just using tf.constant, it needs to be initialized once you use tensorflow
init = tf.initialize_all_variables()

# "With tf.Session() as sess" means we name tf.Session() as sess for the following part
with tf.Session() as sess:
    # we need to initialize tensorflow system before we use it
    sess.run(init)

    # train 201 times, and show the value of Weights and biases
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print("Step: {}, Weights: {}, biases: {}".format(
                step, sess.run(Weights), sess.run(biases)))
