import numpy as np
from scipy.stats import truncnorm
from scipy.special import expit as activation_function


def truncated_normal(mean, sd, low, upp):
    """
    Normal distribution
    """
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class NeuralNetwork:

    def __init__(self, input_layer, output_layer, hidden_layer, learning_rate):
        """
        @param input_layer: number of neurons in input layer
        @param output_layer: number of neurons in output layer
        @param hidden_layer: number of neurons in hidden layer
        @param learning_rate: learning rate in backpropagation
        """
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.hidden_layer = hidden_layer
        self.learning_rate = learning_rate
        self.create_weight_matrices()

    def create_weight_matrices(self):
        """
        Initializing the weights by normal distribution
        """
        rad = 1 / np.sqrt(self.input_layer)
        x = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_layer = x.rvs((self.hidden_layer, self.input_layer))

        rad = 1 / np.sqrt(self.hidden_layer)
        x = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_output_layer = x.rvs((self.output_layer, self.hidden_layer))

    def train(self, input_vector, target_vector):
        """
        Backpropagation
        @param input_vector: Input vector
        @param target_vector: Target vector
        """
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        output_vector1 = np.dot(self.weights_hidden_layer, input_vector)
        output_hidden = activation_function(output_vector1)

        output_vector2 = np.dot(self.weights_output_layer, output_hidden)
        output_network = activation_function(output_vector2)

        output_errors = target_vector - output_network

        tmp = output_errors * output_network * (1.0 - output_network)
        tmp = self.learning_rate * np.dot(tmp, output_hidden.T)
        self.weights_output_layer += tmp

        hidden_errors = np.dot(self.weights_output_layer.T, output_errors)

        tmp = hidden_errors * output_hidden * (1.0 - output_hidden)
        self.weights_hidden_layer += self.learning_rate * np.dot(tmp, input_vector.T)

    def run(self, input_vector):
        """
        Inference
        @param input_vector: input vector
        return: output vector
        """
        input_vector = np.array(input_vector, ndmin=2).T
        output_vector = np.dot(self.weights_hidden_layer, input_vector)
        output_vector = activation_function(output_vector)

        output_vector = np.dot(self.weights_output_layer, output_vector)
        output_vector = activation_function(output_vector)

        return output_vector

    def evaluate(self, data, labels):
        """
        @param data: result
        @param labels: expected result
        return: numbers correct and wrong answers
        """
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs
