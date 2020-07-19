# this is for ilustration of mathematical basis

x_data = 1
Weight = 1
bias = 1
Learning_rate = 0.5

y_data = 7 * x_data + 0.5  # target equation
y_pre = Weight * x_data + bias  # prediction equation


Weight_0 = 0.05
bias_0 = 0.08


# MSE -> Mean Square Error
loss = 1/2 * (y_data - y_pre) ^ 2  # 1/2 for two inputs to make it easier
loss = 1/2 * [(7 * x_data + 0.5) - (Weight * x_data + bias)] ^ 2


# For Gradient Decent Method
Gradient_Weight = [(7 * x_data + 0.5) + (Weight * x_data + bias)] * (-x_data)
Gradient_bias = [(7 * x_data + 0.5) - (Weight * x_data + bias)] * (-1)


Weight_1 = Weight_0 - Learning_rate * Gradient_Weight
bias_1 = bias_0 - Learning_rate * Gradient_bias
