import tensorflow as tf
import numpy as np

# Using linspace to build data with the shape of 300 rows, 1 col
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# placeholder allocates memory space
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])


def add_layer(inputs, in_size, out_size, activation_function=None):
    # Tensorflow recommands that beginning value better not to be zero
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)  # zeros(幾列,幾行)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# ================================== Building Neural Network ===================================

# we are going to build a Neural Network with three layers, which are input_layer, hidden_layer, output_layer in this practise.
# Neurals inside three layers are 1, 10, 1 respectively.
# For the building part, we are just set the structure or build the space for each layer.
# Data or value will be inserted to each layer in the training part.
# P.S.: all the layers between input & output layer are called hidden layer, no matter how many are hidden layers.


l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# Argument: add_layer(inputs,in_size,out_size,activation_function=None)
# For layer_1, input will be the data that we build in the line 4,

prediction = add_layer(l1, 10, 1, activation_function=None)
# you can try to use activation_function in this layer and compare with the different results
# Both ways should be able to predict the value properly, because this model is quite simple


loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))
# using RMS(Root Mean Square) as the method of loss
# reduction_indices introduction: https://blog.csdn.net/qq_33096883/article/details/77479766

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# Choose GradientDescent as the Optimizer with the learning rate of 0.1 to minimize the loss

# ================================== Building Neural Network ===================================

# ================================== Training ===================================

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1001):

        # We feed data "x_data" to the placeholder "xs" to predict the value, and also feed the correct answer "y_data" to train the model
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

        if i % 50 == 0:
            # print the training step and the loss
            print("Training step: ", i, "loss: ", sess.run(
                loss, feed_dict={xs: x_data, ys: y_data}))

# ================================== Training ===================================
