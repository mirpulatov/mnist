MNIST Handwritten Digit Classifier
======

MNIST Handwritten Digit Classifier using numpy without any ML libraries.

### Architecture: Fully-Connected

We use 3 layers: input layer, hidden layer, output layer.

Input layer - 784 neurons.

Hidden layer - 300 neurons.

Output layer - 10 neurons.

### Activation function:

Sigmoid: `1 / 1 + e ^ (-x)`

### Training:

We use backpropogation algorithm (gradient descent algorithm).

### Weights:

We use scipy function *truncnorm* to initialize the weight by normal distribution.

### Requirements:

scipy==1.1.0

numpy==1.15.1

pickle

`pip install -r requirements`

### Result:
After 10 epochs neural network gives as a result:

`Epoch 10:

Accuracy on training set:  0.9908666666666667

Accuracy on test set:  0.975`

