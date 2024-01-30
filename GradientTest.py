import numpy as np

class GradTest:
    def __init__(self, func, grad, i):
        self.eps = 0.5 ** i * 0.001
        self.func = func
        self.d_vec = np.random.random()  # Corrected random number generation
        self.grad = grad

    def gradient_test(self, x):
        f_x = self.func(x)  # Corrected function call
        f_x_eps_d = self.func(x + self.eps * self.d_vec)
        eps_d_transpose = self.eps * self.d_vec  # Removed .T, as it's a 1D array
        O_e = abs(f_x - f_x_eps_d)
        O_e2 = abs(f_x_eps_d - f_x - np.dot(eps_d_transpose, self.grad(x)))  # Corrected np.dot

        return O_e, O_e2
    # Karen
    # make Layer work in NN
    # train feedforward back in NN
    # gradient test

    # Pamela
    # Jacobian test

    # both read about residual network and how forward/back works
