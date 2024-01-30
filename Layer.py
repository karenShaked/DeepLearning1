# Layer Object
import numpy as np
import Activation

class Layer:
    def __init__(self, weightsArray, biasArray, activation):
        """
        :param weightsArray: m*n where
        m - number of neurons in the next layer
        n - number of neurons in last layer
        :param biasArray: m*1 vector
        :param activation: activation function of this layer
        """
        self.weights = weightsArray
        self.biases = biasArray
        self.activation = activation
        self.input = None
        self.wxb = None

    def forward(self, input):
        """
        calculates feedforward for next levels
        :param input: n*1  input vector
        :return: m*1 output vector
        """
        self.input = input
        self.wxb = np.dot(self.weights, input) + self.biases
        layer_result = self.activation.apply(self.wxb)
        return layer_result

    def backward(self, next_layers_gradient , learning_rate):
        """
        given the error derivative by the layer,
        we need to calculate the derivative of the layer
        by weights and biases, and UPDATE them
        :param next_layers_gradient: m*1 gradient from next layers
        :param learning_rate:
        :return: n*1 gradient for former layers
        """
        activation_deriv = self.activation.apply_derivative(self.wxb)
        activation_grad = np.multiply(activation_deriv, next_layers_gradient)
        w_grad = np.dot(activation_grad, self.input.T)
        n = len(self.weights)
        x_grad = 1/n * np.dot(self.weights.T, activation_grad)
        self.weights -= learning_rate * w_grad
        self.biases -= learning_rate * activation_grad
        return x_grad
