import numpy as np

#input data| matrix size (2,1)
a1 = np.array([[4],
              [5]])

# expected output
y_hat = np.array([[8]])

#first layer| weights matrix size (2,2)
w1 = np.array([[1,-4],
              [5,7]])
#first layer| bias matrix size (2,1)
b1 = np.array([[2],
              [5]])
#second layer| weights matrix size (1,2)
w2 = np.array([[3,6]])
#second layer| bias matrix size (1,1)
b2 = np.array([[7]])

#learning rate
lr = 0.005

#helper functions
def ReLu(x):
    a = np.array(x,copy=True)
    a[a<0] = 0
    return a

def ReLu_derivative(x):
    a = np.array(x,copy=True)
    a[a<=0] = 0
    a[a>0] = 1
    return a

def MSE(input, target):
    difference = input - target
    MSE = np.mean(np.square(difference))
    return MSE

#input neurons @ first layer weights + first layer biases = hidden layer neurons| (2,2) @ (2,1) + (2,1) = (2,1)
z2 = w1 @ a1 + b1 #before activation, if less than 0, ReLu would block this neuron
a2 = ReLu(z2) #after activation

#hidden neurons @ second layer weights + second layer biases = output layer neurons| (1,2) @ (2,1) + (1,1) = (1,1)
z3 = w2 @ a2 + b2 #before activation value, if less than 0, ReLu would block this neuron
a3 = ReLu(z3) #after activation

print (a3)
MSError = MSE(a3,y_hat)

#trainning loop
while (MSError > 0.005):
    #input neurons @ first layer weights + first layer biases = hidden(second) layer neurons| (2,2) @ (2,1) + (2,1) = (2,1)
    z2 = w1 @ a1 + b1 #before activation, if less than 0, ReLu would block this neuron
    a2 = ReLu(z2) #after activation

    #hidden neurons @ second layer weights + second layer biases = output(third) layer neurons| (1,2) @ (2,1) + (1,1) = (1,1)
    z3 = w2 @ a2 + b2 #before activation value, if less than 0, ReLu would block this neuron
    a3 = ReLu(z3) #after activation

    print (a3)

    MSError = MSE(a3,y_hat)

    #error of the last(third) layer neurons (the 3rd layer) wrt. pre-activation of the last(third) layer neurons
    gradient_pre3 = (a3 - y_hat) * ReLu_derivative(z3)
    #derivative of weights between 3rd layer neurons and 2nd layer neurons (2nd layer weights)
    w2_derivative = gradient_pre3 @ a2.T
    b2_derivative = gradient_pre3

    #error of the second layer neurons wrt. pre-activation of the last(third) layer neurons
    gradient_pre2 = w2.T @ gradient_pre3 * ReLu_derivative(z2)
    #derivative of weights between 2nd layer neurons and 1st layer neurons (1st layer weights)
    w1_derivative = gradient_pre2 @ a1.T
    b1_derivative = gradient_pre2

    #weights, biases update
    w2.T
    w1.T
    w2 = w2 - lr * w2_derivative
    b2 = b2 - lr * b2_derivative
    w1 = w1 - lr * w1_derivative
    b1 = b1 - lr * b1_derivative