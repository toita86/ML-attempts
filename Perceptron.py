import math
import random

tresh = 0
lam = 0.3
loss_f = 0
new_loss = 0
weights = []

def stable_sigmoid(x):

    if x >= 0:
        z = math.exp(-x)
        sig = 1 / (1 + z)
        return sig
    else:
        z = math.exp(x)
        sig = z / (1 + z)
        return sig

def ReLU(x):
	return max(0.0, x)

def weights_init(x_train, random_init = True):
    global weights
    if random_init == True:
        weights = [(random.random()) for x in range(len(x_train[1]))]
    else:
        weights = [(0.0) for x in range(len(x_train[1]))]

def perceptron(x_inputs):
    global weights
    weighted_sum = 0
    for y in range(len(x_inputs)):
        for k in range(len(weights)):
            weighted_sum += weights[k]*x_inputs[y] + weights[k]*x_inputs[y]
    return stable_sigmoid(weighted_sum + tresh)
    #return ReLU(weighted_sum + tresh)

def training(x_train, x_expected, epochs, early_stop_flag=True):
    global tresh
    global loss_f
    global new_loss
    global weights
    for epoch in range(epochs):
            for y in range(len(x_train)):
                for k in range(len(weights)):
                    weights[k] += lam * (x_expected[y] - perceptron(x_train[y]))*x_train[y][k]
                tresh += lam * (x_expected[y] - perceptron(x_train[y]))
                new_loss += abs(x_expected[y] - perceptron(x_train[y]))
            if early_stop_flag:
                if abs(loss_f - new_loss) > 0.0000000005:
                    loss_f = new_loss
                else:
                    return epoch + 1
    return epochs
