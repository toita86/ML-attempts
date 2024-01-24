'''
Teaching a perceptron an AND gate or an OR gate
'''
import math
import random

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

def perceptron(w,x_train):
    weighted_sum = 0
    for y in range(len(x_train)):
        for k in range(len(weights)):
            weighted_sum += w[k]*x_train[y] + w[k]*x_train[y]
    #return stable_sigmoid(weighted_sum + tresh)
    return ReLU(weighted_sum + tresh)

def training(x_train, x_expected, epochs, early_stop_flag=True):
    global tresh
    global loss_f
    global new_loss
    for epoch in range(epochs):
            for y in range(len(x_train)):
                for k in range(len(weights)):
                    weights[k] += lam * (x_expected[y] - perceptron(weights, x_train[y]))*x_train[y][k]
                tresh += lam * (x_expected[y] - perceptron(weights, x_train[y]))
                new_loss += abs(x_expected[y] - perceptron(weights, x_train[y]))
            if early_stop_flag:
                if abs(loss_f - new_loss) > 0.0000000005:
                    loss_f = new_loss
                else:
                    return epoch + 1
    return epochs

x_train_AND = [[1,1], [1,0], [0,1], [0,0]]
x_train_OR = [[1,1], [1,0], [0,1], [0,0]]

x_expected_AND = [1, 0, 0, 0]
x_expected_OR = [1, 1, 1, 0]

x_test = [[1,1], [1,0], [0,1], [0,0]]

tresh = 0
lam = 0.3
loss_f = 0
new_loss = 0

weights = []

weights_init(x_train_AND, True)
print(f"Before training weights: {weights}")

print ("Before training: ")
for tests in range(4):
    print (f"{x_test[tests]} => {perceptron(weights,x_test[tests])}") 

print(f"Epoch trained: {training(x_train_AND, x_expected_AND, 5000, True)}")

print(f"After training weights: {weights}")
print ("After training results: ")
for tests in range(4):
    #rounded numbers
    #print (f"{x_test[tests]} => {round(perceptron(weights,x_test[tests]))}")
    print (f"{x_test[tests]} => {perceptron(weights,x_test[tests])}") 
