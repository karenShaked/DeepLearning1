from Layer import Layer
import numpy as np
import Activation
from Loss import Loss


class NeuralNetwork:

    def __init__(self, data_dimension,  labels_len, layer_len, hidden_activation, loss_name, final_activation,
                 ResNet=False):
        self.data_dimension = data_dimension
        self.labels_len = labels_len
        self.layer_len = layer_len
        self.layers = []
        input_dim = data_dimension
        output_dim = data_dimension
        for i in range(layer_len):
            output_dim = data_dimension * 5
            self.layers.append(Layer(hidden_activation, input_dim, output_dim))
            input_dim = output_dim
        self.loss_layer = Loss(loss_name, final_activation, output_dim, labels_len)
        self.batch_size = None

    def feedforward(self, input_data):
        """
        :param input_data: data_dimension * batch_size
        :return:
        """
        x_i = input_data
        for layer in self.layers:
            output = layer.forward(x_i)
            x_i = output
        output = self.loss_layer.forward(x_i)
        return output

    def success_percentage(self, y_true, output):
        predicted_labels_arr = np.argmax(output, axis=1)
        true_labels_arr = np.argmax(y_true, axis=0)
        success_count = np.sum(predicted_labels_arr == true_labels_arr)
        success_percentage = (success_count / self.batch_size) * 100
        return success_percentage

    def backpropagation(self, d_theta, original_theta_loss, dx_loss, jac_test):
        all_theta_grad = d_theta
        all_theta_orig = original_theta_loss
        next_layers_gradient = dx_loss
        for i in reversed(range(len(self.layers))):
            next_layers_gradient, theta_i_grad, original_theta_i =\
                self.layers[i].backward(next_layers_gradient, jac_test)
            all_theta_grad = np.concatenate((all_theta_grad, theta_i_grad), axis=0)
            all_theta_orig = np.concatenate((all_theta_orig, original_theta_i), axis=0)
        return all_theta_orig, all_theta_grad

    def train(self, learning_rate, data_matrix, y_true, grad_test, jac_test):
        y_pred = self.feedforward(data_matrix)
        success_percentage = self.success_percentage(y_true, y_pred)
        loss = self.loss_layer.get_loss(y_true)
        dx_loss, d_theta, original_theta_loss = self.loss_layer.calculate_gradients(y_true, grad_test)
        all_theta_old, all_theta_grad = self.backpropagation(d_theta, original_theta_loss, dx_loss, jac_test)
        theta_new = self.sgd_update_theta(learning_rate, all_theta_old, all_theta_grad)
        self.update_theta_layers(theta_new)
        return success_percentage, loss, y_pred

    def test(self, data_matrix, y_true):
        y_pred = self.feedforward(data_matrix)
        success_percentage = self.success_percentage(y_true, y_pred)
        loss = self.loss_layer.get_loss(y_true)
        return success_percentage, loss

    def sgd_update_theta(self, learning_rate, theta_old, theta_grad):
        theta_new = theta_old - learning_rate * theta_grad
        return theta_new

    def update_theta_layers(self, new_theta):
        new_theta = self.loss_layer.update_theta(new_theta)
        for i in reversed(range(len(self.layers))):
            new_theta = self.layers[i].update_theta(new_theta)

    def sgd_select_batch(self, dataset, labels_set):
        self.batch_size = len(dataset[0]) // 2
        random_indexes = sorted(np.random.choice(range(len(dataset[0])), self.batch_size, replace=False))
        selected_datasets = dataset[:, random_indexes]
        selected_labels = labels_set[:, random_indexes]
        return selected_datasets, selected_labels



