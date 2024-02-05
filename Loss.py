from enum import Enum
import numpy as np
from Activation import Activation


class LossFunctions(Enum):
    # y_true and y_pred are the same dimensions -> (label_len, batch_size)
    LEAST_SQUARES = ("Least Squares", "Any",
                     lambda y_true, y_pred: np.mean((y_pred - y_true) ** 2),
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
        self.weights = np.random.uniform(-0.5, 0.5, (label_dim, input_dim))
        # self.weights = np.array([[0.5, 0.75, 1], [0.2, -0.1, 0.5]])
        self.biases = np.random.uniform(-0.5, 0.5, (1, label_dim))
        # self.biases = np.array([0, 2])
        self.batch_size = 1
        self.loss_name = loss_name
        self.activation_name = activation_name

    def _get_compatible_activation(self, loss_name):
        for loss in LossFunctions:
            if loss.description == loss_name:
                return loss.compatible_activation
        raise ValueError(f"Loss function '{loss_name}' not found.")

    def forward(self, X):
        self.batch_size = X.shape[1]
        self.input = np.transpose(X)
        xw_t = np.dot(self.input, np.transpose(self.weights))
        self.Z = xw_t + np.tile(self.biases, (self.batch_size, 1))
        self.output = self.activation.apply(self.Z)
        return self.output

    def get_loss(self, y_true):
        self.output = np.transpose(self.output)
        loss = self.loss_function(y_true, self.output)
        return loss

    def calculate_gradients(self, y_true, grad_test=False):

        if self.loss_name == "Cross Entropy" and self.compatible_activation == "softmax":
            # For softmax + cross-entropy, the gradient simplifies to (output - y_true)
            dZ = self.output - y_true
        else:
            d_loss_output = self.loss_derivative(y_true, self.output)
            d_activation_z = self.activation.apply_derivative(self.Z)
            dZ = np.multiply(np.transpose(d_activation_z), d_loss_output)
            # we want dZ to be in dimensions -> (label_len, batch_size)

        # input dimensions -> (batch_size, input_features)
        dW = np.dot(dZ, self.input) / self.input.shape[0]
        # dW dimensions -> (labels_len, input_features)
        db = np.mean(dZ, axis=1)
        # db dimensions -> (labels_len, 1)
        d_theta = np.concatenate((dW.flatten(), db.flatten()))
        original_theta = np.concatenate((self.weights.flatten(), self.biases.flatten()))
        dX = np.dot(np.transpose(dZ), self.weights)
        # dX dimensions -> (batch_size, input_features)
        if grad_test:
            self.grad_tests_w_x_b(dW, db, dX, y_true)

        return np.transpose(dX), d_theta, original_theta

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

    def grad_tests_w_x_b(self, grad_w, grad_b, grad_x, y_true):
        from GradientTest import GradTest
        test_grad_w = GradTest(GradTest.func_by_loss_w(
            self.loss_name, self.activation_name, self.input, self.biases, y_true),
                                self.weights)
        test_grad_b = GradTest(GradTest.func_by_loss_b(
            self.loss_name, self.activation_name, self.weights, self.input, y_true),
                                self.biases)
        """test_grad_x = GradTest(GradTest.func_by_loss_x(
            self.loss_name, self.activation_name, self.weights, self.biases),
                               self.label_dim, self.input)  # TODO :: check real dimension"""
        i = 10
        test_grad_w.gradient_test(i, grad_w)
        test_grad_b.gradient_test(i, grad_b)


