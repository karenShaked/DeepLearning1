import numpy as np
import matplotlib.pyplot as plt
from Activation import Activation
import Loss


class GradTest:
    @staticmethod
    def func_by_b(activation, weights, input_value):
        return lambda b: activation.apply(np.dot(weights, input_value) + b)

    @staticmethod
    def func_by_w(activation, input_value, biases):
        return lambda w: activation.apply(np.dot(w, input_value) + biases)

    @staticmethod
    def func_by_x(activation, weights, biases):
        return lambda x: activation.apply(np.dot(weights, x) + biases)

    @staticmethod
    def func_by_loss_x(loss, activation, weights, biases):
        func_x = GradTest.func_by_x(activation, weights, biases)
        return lambda x: loss(func_x(x))

    @staticmethod
    def func_by_loss_w(loss_name, activation_name, input_value, biases, y_true):
        batch_size = input_value.shape[0]
        activation = Activation(activation_name)
        loss, loss_derive = Loss.get_loss_function(loss_name)

        def func_w_(w):
            Z = np.dot(input_value, np.transpose(w)) + np.tile(biases, (batch_size, 1))
            output = activation.apply(Z)
            return np.transpose(output)
        return lambda w: loss(y_true, func_w_(w))

    @staticmethod
    def func_by_loss_b(loss_name, activation_name, input_value, weights, y_true):
        batch_size = input_value.shape[0]
        activation = Activation(activation_name)
        loss, loss_derive = Loss.get_loss_function(loss_name)

        def func_b_(b):
            Z = np.dot(input_value, np.transpose(weights)) + np.tile(b, (batch_size, 1))
            output = activation.apply(Z)
            return np.transpose(output)

        return lambda b: loss(y_true, func_b_(b))

    def __init__(self, func, dim, input_val):
        self.eps0 = 1
        self.eps_i = 0.5
        self.func = func
        random_vec = np.random.rand(dim)
        norm = np.linalg.norm(random_vec)
        self.d_vec = random_vec / norm
        self.input = input_val

    def gradient_test(self, iterations, grad_x):
        O_e = []
        O_e2 = []
        for i in range(iterations):
            f_x = self.func(self.input)
            x_eps_d = self.input + (self.eps0 * self.eps_i ** i) * self.d_vec
            f_x_eps_d = self.func(x_eps_d)
            eps_d_transpose = self.eps0 * self.d_vec
            O_e.append(abs(f_x - f_x_eps_d))
            O_e2.append(abs(f_x_eps_d - f_x - np.dot(eps_d_transpose, grad_x)))
        self._plot_grad(O_e, O_e2)
        check_grad = self._check_grad(O_e, O_e2)
        return check_grad

    def _plot_grad(self, O_e, O_e2):
        x = np.arange(0, len(O_e), 0.01)
        plt.plot(x, O_e, label='O(eps) Linear Decrease')
        plt.plot(x, O_e2, label='O(eps^2) Quadratic Decrease')
        plt.legend()
        plt.show()

    def _check_grad(self, O_e, O_e2):
        is_linearly_decreasing = all(O_e[i] * 2 == O_e[i - 1] for i in range(1, len(O_e)))
        is_quadratically_decreasing = all(O_e2[i] * 4 == O_e2[i - 1] for i in range(1, len(O_e2)))
        return is_linearly_decreasing and is_quadratically_decreasing



