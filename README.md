# Perceptron Training for AND and OR Gates

This Python script demonstrates the training of a perceptron to function as an AND gate or an OR gate. The perceptron is a simple neural network unit that can learn to make binary decisions based on input data.

## Requirements

Ensure you have Python installed on your system. The script uses the `math` library for mathematical operations.

## Usage

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. Run the script:

   ```bash
   python perceptron.py
   ```

The script trains the perceptron to mimic the behavior of an AND gate or an OR gate, depending on the training dataset provided.

## Code Overview

- `stable_sigmoid(x)`: Function for a stable implementation of the sigmoid activation function.
- `ReLu(x)`: Rectified Linear Unit (ReLU) activation function.
- `x_train_AND` and `x_train_OR`: Training datasets for AND and OR gates, respectively.
- `x_test`: Test dataset for evaluating the trained perceptron.
- `tresh` and `lam`: Threshold and learning rate parameters.
- `weights`: Initial weights for the perceptron.
- `pp(w, x_train)`: Perceptron function.
- Training the perceptron using the `training_perceptron` function with the specified number of epochs.

## Results

After training, the script prints the weights and the results of the perceptron for the test dataset.

Feel free to experiment with different training datasets or hyperparameters to observe how the perceptron learns different logic gates.