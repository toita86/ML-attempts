import random

import Perceptron as per

x_train_AND = [[1,1], [1,0], [0,1], [0,0]]
x_train_OR = [[1,1], [1,0], [0,1], [0,0]]

x_expected_AND = [1, 0, 0, 0]
x_expected_OR = [1, 1, 1, 0]

x_test = [[1,1], [1,0], [0,1], [0,0]]

per.weights_init(x_train_AND, True)
print(f"Before training weights: {per.weights}")

print ("Before training: ")
for tests in range(4):
    print (f"{x_test[tests]} => {per.perceptron(x_test[tests])}") 

print(f"Epoch trained: {per.training(x_train_AND, x_expected_AND, 5000, True)}")

print(f"After training weights: {per.weights}")
print ("After training results: ")
for tests in range(4):
    #rounded numbers
    #print (f"{x_test[tests]} => {round(per.perceptron(weights,x_test[tests]))}")
    print (f"{x_test[tests]} => {per.perceptron(x_test[tests])}") 

# Dataset for Perceptron Binary Classification Problem

# Features (X): Two-dimensional points
# Labels (y): Binary labels (0 or 1)

# Class 0
class_0 = [
    [2, 3],
    [1, 2],
    [2, 2.5],
    [1.5, 1.8],
    [3, 2.7],
    [2, 3.5],
]

# Class 1
class_1 = [
    [5, 8],
    [6, 6],
    [7, 7],
    [8, 6],
    [6, 7],
    [7, 9],
]

# Combine features and labels
X = class_0 + class_1
y = [0] * len(class_0) + [1] * len(class_1)

# Shuffling the dataset
combined_data = list(zip(X, y))
random.shuffle(combined_data)
X, y = zip(*combined_data)

# Print the dataset
print("Features (X):")
for features in X:
    print(features)

print("\nLabels (y):")
print(y)

per.weights_init(X, True)
print(f"Before training weights: {per.weights}")

print ("Before training: ")
for tests in X:
    print (f"{tests} => {per.perceptron(tests)}") 

print(f"Epoch trained: {per.training(X, y, 500000, True)}")

print(f"After training weights: {per.weights}")
print ("After training results: ")
for tests in X:
    #rounded numbers
    #print (f"{tests} => {round(per.perceptron(weights,tests))}")
    print (f"{tests} => {per.perceptron(tests)}") 