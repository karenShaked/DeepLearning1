import numpy as np
import matplotlib.pyplot as plt
from Activation import Activation
import Loss
import ResLoss


class GradTest:
    @staticmethod
    def func_by_loss_b(loss_name, activation_name, input_value, weights, y_true, ResNet=False):
        """
        input dims - (input_dim, batch_size)
        """
        batch_size = input_value.shape[1]
        activation = Activation(activation_name)
        if ResNet:
            loss = ResLoss.get_loss_function(loss_name)
        else:
            loss = Loss.get_loss_function(loss_name)

        def func_b_(b):
            xt_w = np.dot(np.transpose(input_value), weights)
            z = xt_w + np.tile(b, (batch_size, 1))
            if loss_name == "Cross Entropy":
                output = activation.apply(z)
            else:
                output = z
            if ResNet:
                output += np.transpose(input_value)
            return output

        return lambda b: loss(y_true, func_b_(b))

    @staticmethod
    def func_by_loss_x(loss_name, activation_name, input_value, biases, weights, y_true, ResNet=False):
        """
        input dims - (input_dim, batch_size)
        """
        batch_size = input_value.shape[1]
        activation = Activation(activation_name)
        if ResNet:
            loss = ResLoss.get_loss_function(loss_name)
        else:
            loss = Loss.get_loss_function(loss_name)

        def func_x_(x):
            xt_w = np.dot(np.transpose(x), weights)
            z = xt_w + np.tile(biases, (batch_size, 1))
            if loss_name == "Cross Entropy":
                output = activation.apply(z)
            else:
                output = z
            if ResNet:
                output += np.transpose(x)
            return output

        return lambda x: loss(y_true, func_x_(x))

    @staticmethod
    def func_by_loss_w(loss_name, activation_name, input_value, biases, y_true, ResNet=False):
        """
        input dims - (input_dim, batch_size)
        """
        batch_size = input_value.shape[1]
        activation = Activation(activation_name)
        if ResNet:
            loss = ResLoss.get_loss_function(loss_name)
        else:
            loss = Loss.get_loss_function(loss_name)

        def func_w_(w):
            xt_w = np.dot(np.transpose(input_value), w)
            z = xt_w + np.tile(biases, (batch_size, 1))
            if loss_name == "Cross Entropy":
                output = activation.apply(z)
            else:
                output = z
            if ResNet:
                output += np.transpose(input_value)
            return output

        return lambda w: loss(y_true, func_w_(w))

    def __init__(self, func, input_val, name_val):
        self.eps0 = 1
        self.eps_i = 0.5
        self.func = func
        self.col, self.row = input_val.shape[0], input_val.shape[1]
        random_vec = np.random.rand(self.col * self.row)
        norm = np.linalg.norm(random_vec)
        self.d_vec = random_vec / norm
        self.input = input_val
        self.name_val = name_val

    def gradient_test(self, iterations, grad_input):
        O_e = []
        O_e2 = []
        grad_input = grad_input.flatten()
        for i in range(iterations):
            f_x = self.func(self.input)
            x_eps_d = self.input + (self.eps0 * self.eps_i ** i) * self.d_vec.reshape(self.col, self.row)
            f_x_eps_d = self.func(x_eps_d)
            eps_d_transpose = np.transpose((self.eps0 * self.eps_i ** i) * self.d_vec)
            O_e.append(np.log(abs(f_x - f_x_eps_d)))
            O_e2.append(np.log(abs(f_x_eps_d - f_x - np.dot(eps_d_transpose, grad_input))))
        self.plot_grad(O_e, O_e2)

    def plot_grad(self, O_e, O_e2):
        indexes = range(1, len(O_e) + 1)

        plt.figure(figsize=(10, 5))

        plt.plot(indexes, O_e, marker='o', linestyle='-', color='blue', label='O_e1')
        plt.plot(indexes, O_e2, marker='s', linestyle='--', color='red', label='O_e2')

        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title(f"Gradient Test - O(eps) and O(eps^2) of {self.name_val} Values by Index")
        plt.legend()
        plt.grid(True)
        plt.show()


