import Layer as Layer
import numpy as np
import scipy.io
import Activation as Activation
# define all layers for the network weights and biases

def main():
    folder_path = 'HW1_Data'
    file_name = 'GMMData.mat'

    # Load the .mat file
    mat_data = scipy.io.loadmat(f'{folder_path}/{file_name}')
    """train = 'Yt'
    validation = 'Yv'
    input = data[train]
    data_size = input.shape #[m, n]
    Ys = data[validation]
    NeuralNetwork = NeuralNetwork(input, Ys, data_size)"""


