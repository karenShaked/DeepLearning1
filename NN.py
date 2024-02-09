import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

def cross_entropy_loss(y_true, y_pred):
    # Add a small value to prevent log(0)
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[1]

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size, 1))
        self.X = None

    def forward(self, X):
        self.X = X
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def func_by_w(self, X, b, Y):
        return lambda w: cross_entropy_loss(Y, softmax(np.dot(w, X) + b))

    def func_by_x(self, w, b, Y):
        return lambda x: cross_entropy_loss(Y, softmax(np.dot(w, x) + b))

    def backward(self, X, Y, learning_rate):
        m = Y.shape[1]
        dZ2 = self.A2 - Y  # For cross-entropy loss with softmax, this simplifies to A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, self.A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dX2 = np.dot(self.W2.T, dZ2)
        from GradientTest import GradTest
        test_grad_x = GradTest(self.func_by_x(self.W2, self.b2, Y), self.A1, "X")
        test_grad_w = GradTest(self.func_by_w(self.A1, self.b2, Y), self.W2, "W")
        i = 8
        test_grad_x.gradient_test(i, dX2)
        test_grad_w.gradient_test(i, dW2)
        dZ1 = dX2 * relu_derivative(self.Z1)
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.W2 -= learning_rate * dW2
        self.b1 -= learning_rate * db1
        self.b2 -= learning_rate * db2

    def compute_loss(self, Y, Y_hat):
        return cross_entropy_loss(Y, Y_hat)

    def train(self, X_train, Y_train, epochs, learning_rate):
        for i in range(epochs):
            Y_hat = self.forward(X_train)
            loss = self.compute_loss(Y_train, Y_hat)
            self.backward(X_train, Y_train, learning_rate)
            print(f"Epoch {i}, loss: {loss}")

# Example usage
if __name__ == "__main__":
    np.random.seed(1)
    input_size = 5
    hidden_size = 7
    output_size = 10
    epochs = 3
    learning_rate = 0.01  # Adjusted for better convergence with cross-entropy

    X_train = np.random.randn(input_size, 1)  # Example training data
    Y_train = np.zeros((output_size, 1))  # Example training labels
    nn = NeuralNetwork(input_size, hidden_size, output_size)
    nn.train(X_train, Y_train, epochs, learning_rate)
