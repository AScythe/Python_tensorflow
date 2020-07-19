import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



x_data = np.linspace(-1,1,300)[:, np.newaxis] 
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise 
# f(x) = y = x^2 - 0.5 + noise


#################################### Building structure without value ####################################
# placeholder only ask for space
xs = tf.placeholder(tf.float32,[None,1]) #[Row: None, col: 1]
ys = tf.placeholder(tf.float32,[None,1])


def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1) 
    Wx_plus_b = tf.matmul(inputs,Weights) + biases # output function 
    if activation_function is None:
        outputs = Wx_plus_b
    else: 
        outputs = activation_function(Wx_plus_b)
    return outputs




l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)

prediction = add_layer(l1,10,1,activation_function=None)



loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


#################################### Building structure without value ####################################


fig = plt.figure() 
ax = fig.add_subplot(1,1,1)  #fig.add_subplot(row, column, select window)
ax.scatter(x_data, y_data)   
plt.ion() # allow to update window 
plt.show() 


#################################### Input the value to the system we built ####################################

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1001):
        sess.run(train_step, feed_dict = {xs: x_data, ys: y_data}) # train the model
        if i % 50 ==0:
            print("i= ",i," ",sess.run(loss,feed_dict={xs: x_data, ys: y_data}))
            
            try: # I
                ax.lines.remove(lines[0])
            except:
                pass

            prediction_value = sess.run(prediction, feed_dict={xs: x_data})# get prediction_value
            
            lines = ax.plot(x_data, prediction_value,'r-',lw=5)# draw the red line
            plt.pause(0.1)





