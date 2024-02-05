import numpy as np
from Activation import Activation
from GradientTest import GradTest


class ResLayer:
    def __init__(self, activation_name, input_output_dim, mid_dim):
        """
        :param mid_dim: number of neurons in the activation calculation
        :param input_output_dim: number of neurons in input and output
        :param activation_name: name of activation function of this layer
        weights_1: (mid_dim * input_output_dim) weights
        bias_1: (mid_dim * 1) vector
        weights_2: (input_output_dim * mid_dim) weights
        bias_2: (input_output_dim * 1) vector
        """
        self.weights_1 = np.random.uniform(-0.5, 0.5, (mid_dim, input_output_dim))
        self.weights_2 = np.random.uniform(-0.5, 0.5, (input_output_dim, mid_dim))

        self.biases_1 = np.random.uniform(-0.5, 0.5, (mid_dim, 1))
        self.biases_2 = np.random.uniform(-0.5, 0.5, (input_output_dim, 1))

        self.input = None
        self.batch_size = None
        self.input_output_dim = input_output_dim
        self.mid_dim = mid_dim
        self.activation = Activation(activation_name)
        self.calc_wxb = lambda x: (np.dot(self.weights_1, x) + self.biases_1)
        self.layer_func = lambda x: x + np.dot(self.weights_2, self.activation.apply(np.dot(self.weights_1, x)
                                                                                     + self.biases_1)) + self.biases_2
        self.layer_func_derivative = lambda x: self.activation.apply_derivative(x)
        self.layer_result = None
        self.z = None

    def forward(self, input_matrix):
        """
        calculates feedforward for next levels
        :param input_matrix: (input_output_dim * batch_size)  input vector
        :return: (input_output_dim * batch_size) output vector
        """
        self.input = input_matrix
        self.batch_size = input_matrix.shape[0]
        self.z = self.calc_wxb(self.input)
        self.layer_result = self.layer_func(self.input)
        return self.layer_result

    def backward(self, next_layers_gradient, grad_test=False):
        """
        :param next_layers_gradient: (input_output_dim * batch_size) gradient from next layers
        :return: gradient of x of this layer - (input_dim * batch_size)
                 gradient of theta of this layer
                 grad_f_w - (batch_size * input_dim * output_dim) * 1
                 grad_f_b - (batch_size * output_dim * 1) * 1
                 grad_f_theta - (output_dim * ( 1 + input_dim)) * 1
        """
        layer_j = self.layer_func_derivative(self.z)
        grad_f_b1 = np.dot(self.weights_2, np.multiply(layer_j, next_layers_gradient))
        grad_f_w1 = np.dot(grad_f_b1, np.transpose(np.kron(self.input, np.identity(self.input_output_dim))))
        grad_f_w2 = np.kron(np.transpose(self.activation.apply(self.z)), np.identity(self.mid_dim))
        grad_f_x_i = np.dot(np.transpose(self.weights), grad_f_b)
        grad_f_theta = np.concatenate((grad_f_w.flatten(), grad_f_b.flatten()), axis=0)
        original_theta = np.concatenate((self.weights.flatten(), self.biases.flatten()))
        if grad_test:
            self.grad_tests_w_x_b(grad_f_w, grad_f_b, grad_f_x_i)
        return grad_f_x_i, grad_f_theta, original_theta

    def split_theta_w_b(self, params_vector):
        weights_num, biases_num = self.weights.size, self.biases.size
        updated_weights = params_vector[:weights_num]
        updated_biases = params_vector[weights_num:weights_num + biases_num]
        remaining_params = params_vector[weights_num + biases_num:]
        return updated_weights, updated_biases, remaining_params

    def update_theta(self, params_vector):
        updated_weights, updated_biases, remaining_params = self.split_theta_w_b(params_vector)
        self.weights = updated_weights.reshape(self.weights.shape)
        self.biases = updated_biases.reshape(self.biases.shape)
        return remaining_params

    def grad_tests_w_x_b(self, grad_w, grad_b, grad_x):
        test_grad_w = GradTest(GradTest.func_by_w(self.activation, self.input, self.biases),
                               self.output_dim, self.weights)
        test_grad_b = GradTest(GradTest.func_by_b(self.activation, self.weights, self.input),
                               self.output_dim, self.biases)
        test_grad_x = GradTest(GradTest.func_by_x(self.activation, self.weights, self.biases),
                               self.input_dim * self.batch_size, self.input)
        i = 10
        return test_grad_w.gradient_test(i, grad_w) \
            and test_grad_b.gradient_test(i, grad_b) \
            and test_grad_x.gradient_test(i, grad_x)

