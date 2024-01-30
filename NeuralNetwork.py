import Layer
import numpy as np
import Activation



class NeuralNetwork:

    def __init__(self, input_data,  validation_data, layer_num):
        """
        :param input_data:
        :param validation_data:
        :param layer_num:
        """
        self.output = None
        self.input_data = input_data
        self.data_size = len(input_data)
        self.validation_data = validation_data
        self.valid_size = len(validation_data)
        self.layer_num = layer_num
        layer = Layer(Activation("softmax"), self.data_size, self.valid_size)
        self.layers = [layer]


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


