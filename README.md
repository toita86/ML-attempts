# Perceptron Training
This code implements a simple perceptron for binary classification. The perceptron is trained using a gradient descent approach to minimize the loss function.

## Requirements
Ensure you have Python installed on your system. The script uses the `math` library for mathematical operations.

## Usage

1. **How to Run:**
   - Clone the repository to your local machine:
      ```bash
      git clone https://github.com/toita86/ML-attempts.git
      cd ML-attempts
      ```

   - Run the script:
      ```bash
      python3 Dataset.py
      ```
   
   - Modify the `x_train` and `x_expected` arrays in the code to represent your dataset and corresponding labels.
   
   - Adjust hyperparameters such as `lam` (learning rate) and `epochs` in the `training` function as needed.

2. **Description of Functions:** 
   - `stable_sigmoid(x)`: Implements a stable version of the sigmoid activation function.
   - `ReLU(x)`: Implements the Rectified Linear Unit (ReLU) activation function (comment the function you want to use).
   - `weights_init(x_train, random_init=True)`: Initializes the weights for the perceptron.
   - `perceptron(x_inputs)`: Computes the output of the perceptron given input features.
   - `training(x_train, x_expected, epochs, early_stop_flag=True)`: Trains the perceptron using gradient descent.

3. **Important Notes:**
   - Ensure that the input features (`x_train`) and expected outputs (`x_expected`) are correctly defined for your specific problem.


## Info

- **License:**
   - This code is provided under the [GPL-3.0 license](LICENSE).

- **Acknowledgments:**
   - This code is a simple implementation for educational purposes. It may need modification for more complex scenarios or improved performance.