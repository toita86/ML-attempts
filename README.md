# Perceptron Training for AND and OR Gates

This Python script demonstrates the training of a perceptron to function as an AND gate or an OR gate. The perceptron is a simple neural network unit that can learn to make binary decisions based on input data.

## Requirements

Ensure you have Python installed on your system. The script uses the `math` library for mathematical operations.

## Usage

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/toita86/ML-attempts.git
   cd ML-attempts
   ```

2. Run the script:

   ```bash
   python3 Dataset.py
   ```

The script trains the perceptron to mimic the behavior of an AND gate or an OR gate, depending on the training dataset provided.

## Code Overview

- `stable_sigmoid(x)`: Function for a stable implementation of the sigmoid activation function.
- `ReLU(x)`: Rectified Linear Unit (ReLU) activation function.
- `x_train_AND` and `x_train_OR`: Training datasets for AND and OR gates, respectively.
- `x_expected_AND`and `x_expected_OR`: Are the expected values for the realative operations
- `x_test`: Test dataset for evaluating the trained perceptron.
- `tresh` and `lam`: Threshold and learning rate parameters.
- `weights_init`: Initilization of the weights for the perceptron. Based on the dataset it should automatically choose the number of the weights.
- `perceptron(w, x_train)`: Perceptron function.
- Training the perceptron using the `training_perceptron` function with the specified number of epochs and if you want to activate the early stopping of the training.

## Results

After training, the script prints the weights and the results of the perceptron for the test dataset.

Feel free to experiment with different training datasets or hyperparameters to observe how the perceptron learns different logic gates.
