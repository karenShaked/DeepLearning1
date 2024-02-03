from enum import Enum
import numpy as np
from Activation import Activation
from GradientTest import GradTest

class LossFunctions(Enum):
    LEAST_SQUARES = ("Least Squares", "Any",
                     lambda y_true, y_pred: np.mean(0.5 * (y_pred - y_true) ** 2),
                     lambda y_true, y_pred: (y_pred - y_true) / y_true.shape[1])
    CROSS_ENTROPY = ("Cross Entropy", "softmax",
                     lambda y_true, y_pred: -np.mean(
                         y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9)),
                     lambda y_true, y_pred: -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.shape[1])

    @property
    def description(self):
        return self.value[0]

    @property
    def compatible_activation(self):
        return self.value[1]

    @property
    def function(self):
        return self.value[2]

    @property
    def derivative(self):
        return self.value[3]


def get_loss_function(name):
    for loss in LossFunctions:
        if loss.description == name:
            return loss.function, loss.derivative
    raise ValueError(f"Loss function '{name}' not found.")


class Loss:
    def __init__(self, loss_name, activation_name, input_dim, label_dim):
        self.loss_name = loss_name
        self.loss_function, self.loss_derivative = get_loss_function(loss_name)
        self.activation = Activation(activation_name)
        self.compatible_activation = self._get_compatible_activation(loss_name)
        self.input_dim = input_dim
        self.label_dim = label_dim

        if self.compatible_activation != "Any" and self.compatible_activation != activation_name:
            raise ValueError(f"{loss_name} is not compatible with {activation_name} activation.")
        self.Z = None
        self.output = None
        self.input = None
        self.weights = np.random.rand(label_dim, input_dim)
        self.biases = np.random.rand(1, label_dim)
        self.batch_size = 1

    def _get_compatible_activation(self, loss_name):
        for loss in LossFunctions:
            if loss.description == loss_name:
                return loss.compatible_activation
        raise ValueError(f"Loss function '{loss_name}' not found.")

    def forward(self, X):
        self.batch_size = X.shape[1]
        self.input = np.transpose(X)
        self.Z = np.dot(self.input, np.transpose(self.weights)) + np.tile(self.biases, (self.batch_size, 1))
        self.output = self.activation.apply(self.Z)
        return self.output

    def get_loss(self, y_true):
        self.output = np.transpose(self.output)
        loss = self.loss_function(y_true, self.output)
        return loss

    def calculate_gradients(self, y_true):

        if self.loss_name == "Cross Entropy" and self.compatible_activation == "softmax":
            # For softmax + cross-entropy, the gradient simplifies to (output - y_true)
            dZ = self.output - y_true
        else:
            d_activation_output = self.loss_derivative(y_true, self.output)
            dZ = self.activation.apply_derivative(self.Z) * d_activation_output

        dW = np.dot(dZ, self.input) / self.input.shape[0]
        db = np.mean(dZ, axis=1)
        d_theta = np.concatenate((dW.flatten(), db.flatten()))
        original_theta = np.concatenate((self.weights.flatten(), self.biases.flatten()))
        dX = np.dot(np.transpose(dZ), self.weights)

        return dX, d_theta, original_theta

    def update_theta(self, params_vector):
        # Calculate the total number of weight and bias parameters for this layer
        weights_num = self.weights.size
        biases_num = self.biases.size

        # Extract the relevant parameters for weights and biases from the top of params_vector
        updated_weights = params_vector[:weights_num].reshape(self.weights.shape)
        updated_biases = params_vector[weights_num:weights_num + biases_num].reshape(self.biases.shape)

        # Update the weights and biases
        self.weights = updated_weights
        self.biases = updated_biases

        # Return the remaining part of params_vector that was not used for updating this layer
        remaining_params = params_vector[weights_num + biases_num:]

        return remaining_params

    def grad_tests_w_x_b(self, grad_w, grad_b, grad_x):
        test_grad_w = GradTest(GradTest.func_by_loss_w(
            self.loss_function, self.activation, self.input, self.biases),
                               self.label_dim, self.weights)
        test_grad_b = GradTest(GradTest.func_by_loss_b(
            self.loss_function, self.activation, self.weights, self.input),
                               self.label_dim, self.biases)
        test_grad_x = GradTest(GradTest.func_by_loss_x(
            self.loss_function, self.activation, self.weights, self.biases),
                               self.label_dim, self.input)  # TODO :: check real dimension
        i = 10
        return test_grad_w.gradient_test(i, grad_w) \
            and test_grad_b.gradient_test(i, grad_b) \
            and test_grad_x.gradient_test(i, grad_x)


