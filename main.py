from matplotlib import pyplot as plt

from Layer import Layer
import numpy as np
import scipy.io
from Activation import Activation
from NeuralNetwork import NeuralNetwork


def plot_success_loss(indexes, success_loss):
    for name, value in success_loss.items():
        plt.figure(figsize=(10, 5))
        plt.plot(indexes, value, marker='s', linestyle='-', color='purple', label=f"{name}")
        plt.xlabel('iterations')
        plt.ylabel(f"{name} Values")
        plt.title(f"{name} by iterations")
        plt.legend()
        plt.grid(True)
        plt.show()


def main():
    folder_path = 'HW1_Data'
    file_names = ['SwissRollData.mat', 'GMMData.mat', 'PeaksData.mat']

    for file_name in file_names:
        mat_data = scipy.io.loadmat(f'{folder_path}/{file_name}')
        train = 'Yt'
        validation = 'Ct'
        input_data = mat_data[train]
        data_dim = input_data.shape[0]  # [input features, num of data points]
        data_labels = mat_data[validation]
        label_size = data_labels.shape[0]  # [num of labels, num of data points]
        num_of_hidden_layers = 1
        first_model_softmax_regression = NeuralNetwork(data_dim, label_size, num_of_hidden_layers, "sigmoid",
                                                       "Cross Entropy", "softmax")
        #  first_model_softmax_regression = NeuralNetwork(4, 4, 1, "sigmoid", "Least Squares", "softmax")
        #  input_demo = np.array([[1], [2], [3], [1]])  # [input features, num of data points]
        #  labels = np.array([[1], [0], [0], [0]])  # [num of labels, num of data points]

        learning_rate = 0.01
        grad_test = False
        loss_arr, success_arr, index_arr = [], [], []
        for i in range(21):
            selected_data, selected_labels = first_model_softmax_regression.sgd_select_batch(input_data, data_labels)
            success_percentage, loss = first_model_softmax_regression.train(learning_rate, selected_data, selected_labels,
                                                                            grad_test)
            if i % 4 == 0:
                print(f"loss {i} - {loss} \nsuccess percentage:{success_percentage}")
                loss_arr.append(loss)
                success_arr.append(success_percentage)
                index_arr.append(i)
        plot_success_loss(index_arr, {"success percentage": success_arr, "loss": loss_arr})


if __name__ == '__main__':
    main()

