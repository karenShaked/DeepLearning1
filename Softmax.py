# softmax
softmax = """
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
"""

# softmax_loss
softmax_loss = """
import numpy as np

def softmax_loss(W, X, y, reg):
    
    Softmax loss function, similar to cross-entropy."""

# gradient testing
# Generate some synthetic data for testing
"""
num_samples = 100
num_features = 10
num_classes = 5

X = np.random.randn(num_samples, num_features)
y = np.random.randint(num_classes, size=num_samples)
W = np.random.randn(num_classes, num_features)
b = np.random.randn(num_classes)

# Compute gradients using the Softmax loss function
loss, dW, db = softmax_loss(X, y, W, b)

# Gradient check
epsilon = 1e-5

for i in range(num_classes):
    for j in range(num_features):
        # Perturb the weight and compute loss
        W_perturbed = W.copy()
        W_perturbed[i, j] += epsilon
        loss_perturbed, _, _ = softmax_loss(X, y, W_perturbed, b)

        # Compute numerical gradient
        numerical_gradient = (loss_perturbed - loss) / epsilon

        # Compare with computed gradient
        assert np.isclose(numerical_gradient, dW[i, j], rtol=1e-3)

print("Gradient check passed.")
"""

