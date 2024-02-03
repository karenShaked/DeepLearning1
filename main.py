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
    first_model_softmax_regression = NeuralNetwork(data_dim, label_size, 0, None, "Cross Entropy", "softmax")
    for i in range(10):
        selected_data, selected_labels = first_model_softmax_regression.sgd_select_batch(input_data, data_labels)
        print(f"loss {i+1} - {first_model_softmax_regression.train(1, selected_data, selected_labels)}")


if __name__ == '__main__':
    main()

