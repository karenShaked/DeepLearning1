import numpy as np


class Activation:
    def __init__(self, name):
        self.function_name = name
        self.function, self.derivative = self._get_function_and_derivative(name)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def ReLU(x):
        return np.maximum(0, x)

    @staticmethod
    def ReLU_derivative(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability improvement by subtracting max from x
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @staticmethod
    def softmax_derivative(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability improvement by subtracting max from x
        s = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        d_softmax = (1-s)*s
        return d_softmax

    @classmethod
    def _get_function_and_derivative(cls, name):
        if hasattr(cls, name) and hasattr(cls, name + "_derivative"):
            return getattr(cls, name), getattr(cls, name + "_derivative")
        else:
            return None, None

    def apply(self, x):
        return self.function(x)

    def apply_derivative(self, x):
        return self.derivative(x)
