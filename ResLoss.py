from enum import Enum
import numpy as np
from Activation import Activation
from Loss import LossFunctions, get_loss_function

class ResLoss:
    def __init__(self, loss_name, activation_name, dim):
        self.loss_function = get_loss_function(loss_name)
        self.activation = Activation(activation_name)
        self.dim = dim
        self.Z = None
        self.output = None
        self.input = None
        self.weights = np.random.uniform(-0.5, 0.5, (dim, dim))
        self.biases = np.random.uniform(-0.5, 0.5, (1, dim))
        self.batch_size = 1
        self.loss_name = loss_name
        self.activation_name = activation_name

    def forward(self, x):
        """
        input dims - (dim, batch_size)
        w dims - (dim, dim)
        bias dims - (1, dim)
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        self.batch_size = x.shape[1]
        self.input = x
        # input dims - (dim, batch_size)
        xt_w = np.dot(np.transpose(self.input), self.weights)
        # w dims - (dim, dim)
        # x.T @ w = (batch_size, dim) * (dim, dim) = (batch_size, dim)
        self.Z = xt_w + np.tile(self.biases, (self.batch_size, 1))
        # bias dims - (1, dim)
        # z = x.T @ w + b = (batch_size, dim) + for each row add bias -> (batch_size, dim)
        if self.loss_name == "Cross Entropy":
            self.output = self.activation.apply(np.transpose(x) + self.Z)
        if self.loss_name == "Least Squares":
            self.output = np.transpose(x) + self.Z
        return self.output

    def get_loss(self, y_true):
        y_true = np.transpose(y_true)
        loss = self.loss_function(y_true, self.output)
        return loss

    def calculate_gradients(self, y_true, grad_test=False):
        """
        input dims - (dim, batch_size)
        w dims - (dim, dim)
        bias dims - (1, dim)
        """
        y_true = np.transpose(y_true)
        d_loss_out = (1/self.batch_size)*(self.output - y_true)
        # d_loss_out = (output - y_true dims) -> (batch_size, dim)
        d_x = np.ones((self.dim, self.batch_size)) + np.dot(self.weights, np.transpose(d_loss_out))
        # w dims - (dim, dim)
        # d_x dims = (dim, batch_size) + (dim, dim) * (dim, batch_size) -> (dim, batch_size)
        d_w = np.dot(self.input, d_loss_out)
        # d_w dims = (dim, batch_size) * (batch_size, dim) -> (dim, dim)
        d_b = np.sum(d_loss_out, axis=0)
        # d_b dims = (batch_size, dim) sum over columns -> (1, dim)
        d_theta = np.concatenate((d_w.flatten(), d_b.flatten()))
        original_theta = np.concatenate((self.weights.flatten(), self.biases.flatten()))
        if grad_test:
            self.grad_tests_w_x_b(d_w, d_x, d_b, y_true)
        return d_x, d_theta, original_theta

    def grad_tests_w_x_b(self, grad_w, grad_x, grad_b, y_true):
        from GradientTest import GradTest
        resnet = True
        test_grad_w = GradTest(GradTest.func_by_loss_w(
            self.loss_name, self.activation_name, self.input, self.biases, y_true, resnet), self.weights, "W")
        test_grad_x = GradTest(GradTest.func_by_loss_x(
            self.loss_name, self.activation_name, self.input, self.biases, self.weights, y_true, resnet), self.input, "X")
        test_grad_b = GradTest(GradTest.func_by_loss_b(
            self.loss_name, self.activation_name, self.input, self.weights, y_true, resnet), self.biases, "b")
        i = 10
        test_grad_w.gradient_test(i, grad_w)
        test_grad_x.gradient_test(i, grad_x)
        test_grad_b.gradient_test(i, grad_b)

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




