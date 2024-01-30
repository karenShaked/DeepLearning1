from enum import Enum
import numpy as np


class LossFunctions(Enum):
    LEAST_SQUARES = ("Least Squares",
                     lambda y_true, y_pred: (y_pred - y_true) ** 2,
                     lambda y_true, y_pred: 2 * (y_pred - y_true))
    CROSS_ENTROPY = ("Cross Entropy",
                     lambda y_true, y_pred: -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)),
                     lambda y_true, y_pred: -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred)))

    def __init__(self, name, function, derivative):
        self.name = name
        self.function = function
        self.derivative = derivative


def get_loss_function(name):
    for loss in LossFunctions:
        if loss.name == name:
            return loss.function, loss.derivative
    raise ValueError(f"Loss function '{name}' not found.")


class Loss:
    def __init__(self, loss_name, dimension):
        self.loss_function, self.loss_derivative = get_loss_function(loss_name)
        self.dimension = dimension
        self.y_train = []
        self.y_test = []

    def get_loss(self, y_train, y_test):
        if len(y_train) != self.dimension or len(y_test) != self.dimension:
            raise ValueError(f"Dimension of y_train ({len(y_train)}) and y_test ({len(y_test)})"
                             f" must be equal to {self.dimension}")
        self.y_train = y_train
        self.y_test = y_test
        return [self.loss_function(self.y_test, self.y_train), self.get_gradient]

    def get_gradient(self):
        return self.loss_derivative(self.y_test, self.y_train)
