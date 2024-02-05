from Layer import Layer
import numpy as np
import scipy.io
from Activation import Activation
from NeuralNetwork import NeuralNetwork


def main():
    folder_path = 'HW1_Data'
    file_name = 'SwissRollData.mat'

    # Load the .mat file
    mat_data = scipy.io.loadmat(f'{folder_path}/{file_name}')
    train = 'Yt'
    validation = 'Ct'
    input_data = mat_data[train]
    data_dim = input_data.shape[0]  # [input features, num of data points]
    data_labels = mat_data[validation]
    label_size = data_labels.shape[0]  # [num of labels, num of data points]
    num_of_hidden_layers = 0
    # first_model_softmax_regression = NeuralNetwork(data_dim, label_size, num_of_hidden_layers, None,
    # "Cross Entropy", "softmax")
    first_model_softmax_regression = NeuralNetwork(4, 4, 1, "sigmoid", "Least Squares", "softmax")
    grad_test = True
    input_demo = np.array([[1], [2], [3], [1]])  # [input features, num of data points]
    labels = np.array([[1], [0], [0], [0]])  # [num of labels, num of data points]
    for i in range(1001):
        # selected_data, selected_labels = first_model_softmax_regression.sgd_select_batch(input_data, data_labels)
        loss = first_model_softmax_regression.train(0.5, input_demo, labels, grad_test)
        if i % 100 == 0:
            print(f"loss {i} - {loss}")


if __name__ == '__main__':
    main()

