from enum import Enum
import numpy as np
from Activation import Activation


class LossFunctions(Enum):
    # y_true and y_pred are the same dimensions -> (batch_size, label_len)
    LEAST_SQUARES = ("Least Squares",
                     lambda y_true, y_pred: 0.5 * np.linalg.norm(y_pred - y_true) ** 2 / y_true.shape[0])
    CROSS_ENTROPY = ("Cross Entropy",
                     lambda y_true, y_pred: -np.sum(np.multiply(y_true, np.log(y_pred + 1e-9)))/y_true.shape[0])

    @property
    def description(self):
        return self.value[0]

    @property
    def function(self):
        return self.value[1]


def get_loss_function(name):
    for loss in LossFunctions:
        if loss.description == name:
            return loss.function
    raise ValueError(f"Loss function '{name}' not found.")


class Loss:
    def __init__(self, loss_name, activation_name, input_dim, label_dim):
        self.loss_function = get_loss_function(loss_name)
        self.activation = Activation(activation_name)
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.Z = None
        self.output = None
        self.input = None
        self.weights = np.random.uniform(-0.5, 0.5, (input_dim, label_dim))
        self.biases = np.random.uniform(-0.5, 0.5, (1, label_dim))
        self.batch_size = 1
        self.loss_name = loss_name
        self.activation_name = activation_name

    def forward(self, x):
        """
        input dims - (input_dim, batch_size)
        w dims - (input_dim, labels_dim)
        w dims - (input_dim, labels_dim)
        bias dims - (1, labels_dim)
        """
        self.batch_size = x.shape[1]
        self.input = x
        # input dims - (input_dim, batch_size)
        xt_w = np.dot(np.transpose(self.input), self.weights)
        # w dims - (input_dim, labels_dim)
        # x.T @ w = (batch_size, input_dim) * (input_dim, labels_dim) = (batch_size, labels_dim)
        self.Z = xt_w + np.tile(self.biases, (self.batch_size, 1))
        # bias dims - (1, labels_dim)
        # z = x.T @ w + b = (batch_size, labels_dim) + for each row add bias -> (batch_size, labels_dim)
        if self.loss_name == "Cross Entropy":
            self.output = self.activation.apply(self.Z)
        if self.loss_name == "Least Squares":
            self.output = self.Z
        return self.output

    def get_loss(self, y_true):
        y_true = np.transpose(y_true)
        loss = self.loss_function(y_true, self.output)
        # output dims - (batch_size, labels_dim)
        return loss

    def calculate_gradients(self, y_true, grad_test=False):
        """
        output dims - (batch_size, labels_dim)
        input dims - (input_dim, batch_size)
        w dims - (input_dim, labels_dim)
        bias dims - (1, labels_dim)
        """
        y_true = np.transpose(y_true)
        d_loss_out = (1 / self.batch_size) * (self.output - y_true)
        # d_loss_out = (output - y_true dims) -> (batch_size, labels_dim)
        d_x = np.dot(self.weights, np.transpose(d_loss_out))
        # w dims - (input_dim, labels_dim)
        # d_x dims = (input_dim, labels_dim) * (labels_dim, batch_size) -> (input_dim, batch_size)
        d_w = np.dot(self.input, d_loss_out)
        # d_w dims = (input_dim, batch_size) * (batch_size, labels_dim) -> (input_dim, labels_dim)
        d_b = np.sum(d_loss_out, axis=0)
        # d_b dims = (batch_size, labels_dim) sum over columns -> (1, labels_dim)
        d_theta = np.concatenate((d_w.flatten(), d_b.flatten()))
        original_theta = np.concatenate((self.weights.flatten(), self.biases.flatten()))
        if grad_test:
            self.grad_tests_w_x_b(d_w, d_x, d_b, y_true)
        return d_x, d_theta, original_theta

    def grad_tests_w_x_b(self, grad_w, grad_x, grad_b, y_true):
        from GradientTest import GradTest
        test_grad_w = GradTest(GradTest.func_by_loss_w(
            self.loss_name, self.activation_name, self.input, self.biases, y_true), self.weights, "W")
        test_grad_x = GradTest(GradTest.func_by_loss_x(
            self.loss_name, self.activation_name, self.input, self.biases, self.weights, y_true), self.input, "X")
        test_grad_b = GradTest(GradTest.func_by_loss_b(
            self.loss_name, self.activation_name, self.input, self.weights, y_true), self.biases, "b")
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
