'''
Teaching a perceptron an AND gate or an OR gate
'''
import math

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

def perceptron(w,x_train):
    #return stable_sigmoid(w[0]*x_train[0] + w[1]*x_train[1] + tresh)
    return ReLU(w[0]*x_train[0] + w[1]*x_train[1] + tresh)

def training(x_train, epochs, early_stop_flag=True):
    global tresh
    global loss_f
    global new_loss
    for epoch in range(epochs):
            for y in range(4):
                weights[0] += lam * (x_train[y][2] - perceptron(weights, x_train[y]))*x_train[y][0]*x_train[y][1]
                weights[1] += lam * (x_train[y][2] - perceptron(weights, x_train[y]))*x_train[y][1]*x_train[y][0]
                tresh += lam * (x_train[y][2] - perceptron(weights, x_train[y]))
                new_loss += abs(x_train[y][2] - perceptron(weights, x_train[y]))
            if early_stop_flag:
                if abs(loss_f - new_loss) > 0.0000000005:
                    loss_f = new_loss
                else:
                    return epoch + 1
    return epochs

x_train_AND = [[1,1,1], [1,0,0], [0,1,0], [0,0,0]]
x_train_OR = [[1,1,1], [1,0,1], [0,1,1], [0,0,0]]

x_test = [[1,1], [1,0], [0,1], [0,0]]

tresh = 0
lam = 0.3
loss_f = 0
new_loss = 0

weights = [0.0, 0.0]

print(f"Before training weights: {weights}")
print ("Before training: ")
for tests in range(4):
    print (f"{x_test[tests]} => {perceptron(weights,x_test[tests])}") 

print(f"Epoch trained: {training(x_train_AND, 5000, True)}")

print(f"After training weights: {weights}")
print ("After training results: ")
for tests in range(4):
    #rounded numbers
    #print (f"{x_test[tests]} => {round(perceptron(weights,x_test[tests]))}")
    print (f"{x_test[tests]} => {perceptron(weights,x_test[tests])}") 
