import Layer as Layer
import numpy as np
import Activation as Activation


class NeuralNetwork:

    def __init__(self, input, validation, data_size):
        """

        :param input: initial x vector i
        :param validation:
        :param data_size:
        """
        self.output = None
        self.input = input
        self.validation = validation
        self.data_size = data_size
        self.layers = []  # NEED TO ADD: definition of layers
        # layer1 = Layer([[1, 1, 1], [1, 1, 1], [1, 1, 1]], [2, 2, 2]) #make not a square
        # layer2 = Layer([[1, 1, 1], [1, 1, 1], [1, 1, 1]], [2, 2, 2])
        # layers = [layer1, layer2]

    def train(self):  # process of training the NN
        error = 0
        for x, y in (self.input, self.validation):
            output = self.FeedForward(x)  # passing the data through the NN layers
            error += self.mean_squared_error(output, y)  # computing the error
            gradient = 2 * (y - output) / np.size(output)  # gradient of the loss
            gradient = self.softmax_gradient(gradient, learning_rate)  # first backpropogation for softmax
            for layer in reversed(self.layers):  # Backpropogation - computing the gradient - SGD
                gradient = layer.backward(gradient, learning_rate)  # NEED TO ADD: learning rate

            ##TO BE CONTINUED

    def FeedForward(self, input):
        output = input
        for layer in self.layers:  # need to change so won't affect the last layer
            output = layer.forward(output)
        output = Activation.Softmax(output)
        return output

    def mean_squared_error(m, y_true, y_pred):  #####incorrect calculation
        error = 0
        for Ytrue, Ypred in (y_true, y_pred):
            error += np.sum(np.mean(np.power(Ytrue - Ypred, 2)))
        return (1 / m) * error

    def softmax_gradient(self, output_gradient, learning_rate):  # check if true
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)


#################################################################

def stochastic_gradient_decent(learning_rate, ):
    # softmax activation
    #
    pass


# plot success percentages of the data classification
def plot_success_rate():
    pass


# feedforward
"""
def feedforward(x, weights, biases):
    Feedforward the input x through the neural network
    :param x: input data
    :param weights: weights of the neural network
    :param biases: biases of the neural network
    :return: output of the neural network
    z = np.dot(x, weights) + biases
    return z
"""

# backpropagation function
"""
def backpropagation(x, y, z, weights, biases, learning_rate=0.01, reg=0.0):
    Backpropagation algorithm
    :param x: input data
    :param y: labels
    :param z: output of the neural network
    :param weights: weights of the neural network
    :param biases: biases of the neural network
    :param learning_rate: learning rate
    :param reg: regularization parameter
    :return: weights and biases
    """

# stochastic gradient descent algorithm

"""
import numpy as np

def sgd(W, X, y, learning_rate=0.01, reg=0.0):
    ""Stochastic gradient descent."""
