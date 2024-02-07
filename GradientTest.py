import numpy as np
import matplotlib.pyplot as plt
from Activation import Activation
import Loss


class GradTest:
    @staticmethod
    def func_by_loss_x(loss_name, activation_name, input_value, biases, weights, y_true):
        batch_size = input_value.shape[1]
        activation = Activation(activation_name)
        loss = Loss.get_loss_function(loss_name)

        def func_x_(x):
            xt_w = np.dot(np.transpose(x), weights)
            z = xt_w + np.tile(biases, (batch_size, 1))
            if loss_name == "Cross Entropy":
                output = activation.apply(z)
            else:
                output = z
            return output

        return lambda x: loss(y_true, func_x_(x))

    @staticmethod
    def func_by_loss_w(loss_name, activation_name, input_value, biases, y_true):
        batch_size = input_value.shape[1]
        activation = Activation(activation_name)
        loss = Loss.get_loss_function(loss_name)

        def func_w_(w):
            xt_w = np.dot(np.transpose(input_value), w)
            z = xt_w + np.tile(biases, (batch_size, 1))
            if loss_name == "Cross Entropy":
                output = activation.apply(z)
            else:
                output = z
            return output

        return lambda w: loss(y_true, func_w_(w))

    def __init__(self, func, input_val):
        self.eps0 = 1
        self.eps_i = 0.5
        self.func = func
        self.col, self.row = input_val.shape[0], input_val.shape[1]
        random_vec = np.random.rand(self.col * self.row)
        norm = np.linalg.norm(random_vec)
        self.d_vec = random_vec / norm
        self.input = input_val

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
        plt.plot(indexes, O_e2, marker='s', linestyle='-', color='red', label='O_e2')

        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('O(eps) and O(eps^2) Values by Index')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Print both values in 2 columns
        print("Index\tO_e1\t\t\t\t\tO_e2")
        for i, (val1, val2) in enumerate(zip(O_e, O_e2), 1):
            print(f"{i}\t{val1}\t{val2}")


