import numpy as np
from Activation import Activation
from GradientTest import GradTest


class Layer:
    def __init__(self, activation_name, input_dim, output_dim):
        """
        :param output_dim: number of neurons in the next layer
        :param input_dim: number of neurons in last layer
        :param activation_name: name of activation function of this layer
        weights: (output_dim * input_dim) weights
        bias: (output_dim * 1) vector
        """
        self.weights = np.random.uniform(-0.5, 0.5, (output_dim, input_dim))
        self.biases = np.random.uniform(-0.5, 0.5, (output_dim, 1))
        self.input = None
        self.batch_size = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation_name
        self.activation = Activation(activation_name)
        self.calc_wxb = lambda x: (np.dot(self.weights, x) + self.biases)
        self.layer_func = lambda x: self.activation.apply(np.dot(self.weights, x) + self.biases)
        self.layer_func_derivative = lambda x: self.activation.apply_derivative(x)
        self.layer_result = None
        self.z = None

    def forward(self, input_matrix):
        """
        calculates feedforward for next levels
        :param input_matrix: (input_dim * batch_size)  input vector
        :return: (output_dim * batch_size) output vector
        """
        self.input = input_matrix
        self.batch_size = input_matrix.shape[0]
        self.z = self.calc_wxb(self.input)
        # w dims -> (output_dim, input_dim)
        # x dims -> (input_dim, batch_size)
        # b dims -> (output_dim, 1)
        # z = wx+b ->(output_dim, batch_size)
        self.layer_result = self.layer_func(self.input)
        # layer_result = activation(z) ->(output, batch_size)
        return self.layer_result

    def backward(self, next_layers_gradient, jac_test):
        activation_derivative = self.layer_func_derivative(self.z)
        # activation_derivative = derivative_activation(z) ->(output_dim, batch_size)
        grad_f = np.multiply(activation_derivative, next_layers_gradient)
        # grad_f = layer_j * next_layers_gradient ->(output_dim, batch_size)
        grad_f_w = np.dot(grad_f, np.transpose(self.input))
        # grad_f_w = grad_f @ x.T ->(output_dim, batch_size)*(batch_size, input_dim)->(output_dim, input_dim)
        grad_f_x_i = np.dot(np.transpose(self.weights), grad_f)
        # grad_f_x_i = w.T @ grad_f ->(input_dim, output_dim)*(output, batch_size)->(input, batch_size)
        grad_f_b = grad_f.sum(axis=1).reshape(-1, 1)
        # grad_f_b = sum over columns(grad_f) ->(output, 1)
        grad_f_theta = np.concatenate((grad_f_w.flatten(), grad_f_b.flatten()), axis=0)
        original_theta = np.concatenate((self.weights.flatten(), self.biases.flatten()))
        if jac_test:
            # only for batch_size = 1
            input_one_col = self.input[:, 0:1]
            # input_one_col -> (input_dim, 1)
            activation_derivative_one_col = activation_derivative[:, 0:1]
            # activation_derivative_one_col -> (output_dim, 1)
            jac_w = np.outer(activation_derivative_one_col, input_one_col)
            # [[σ'(z1) * x1, σ'(z1) * x2,...],
            # ....
            # [σ'(zm) * x1,... ]]
            # jac_w -> (output_dim, input_dim)
            # the real jacobian doesn't look exactly like this (it is padded with many zeros),
            # but we made the proper changes to make it right and improve performance
            jac_x = np.multiply(self.weights, np.dot(activation_derivative_one_col, np.ones((1, self.input_dim))))
            # [[σ'(z1) * w11, σ'(z1) * w12 ,...],
            # ...
            # [σ'(zm) * wm1, ... ]]
            # jac_x -> (output_dim, input_dim)
            jac_b = activation_derivative_one_col
            self.jac_tests_w_x(jac_w, jac_x, jac_b, input_one_col)

        return grad_f_x_i, grad_f_theta, original_theta

    def jac_tests_w_x(self, jac_w, jac_x, jac_b, input_x):
        from JacobianTest import JacTest
        test_jac_w = JacTest(JacTest.func_by_w(self.activation_name, input_x, self.biases), self.weights, "w")
        test_jac_x = JacTest(JacTest.func_by_x(self.activation_name, self.weights, self.biases), input_x, "x")
        test_jac_b = JacTest(JacTest.func_by_b(self.activation_name, input_x, self.weights), self.biases, "b")
        i = 10
        test_jac_w.jacobian_test(i, jac_w)
        test_jac_x.jacobian_test(i, jac_x)
        test_jac_b.jacobian_test(i, jac_b)

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
