from matplotlib import pyplot as plt
import numpy as np
import scipy.io
from NeuralNetwork import NeuralNetwork


def success_percentage_graph(success_percentage, index_arr):
    plt.plot(index_arr, success_percentage)
    plt.xlabel('')
    plt.ylabel('success')
    plt.title('success rate')
    plt.show()


def plot_success_loss(indexes, success_loss_train, success_loss_test):
    for (name_train, value_train), (name_test, value_test) in \
            zip(success_loss_train.items(), success_loss_test.items()):
        plt.figure(figsize=(10, 5))
        plt.plot(indexes, value_train, marker='s', linestyle='-', color='purple', label=f"train {name_train}")
        plt.plot(indexes, value_test, marker='s', linestyle='-', color='red', label=f"test {name_test}")
        plt.xlabel('iterations')
        plt.ylabel(f"Values")
        plt.title(f"{name_train} vs {name_test} by iterations")
        plt.legend()
        plt.grid(True)
        plt.show()


def extract_data(file_name, folder_name):
    mat_data = scipy.io.loadmat(f'{folder_name}/{file_name}')
    train = 'Yv'
    labels_train = 'Cv'
    test = 'Yt'
    labels_test = 'Ct'
    input_train = mat_data[train]
    input_test = mat_data[test]
    data_dim = input_train.shape[0]  # [input features, num of data points]
    train_labels = mat_data[labels_train]
    test_labels = mat_data[labels_test]
    label_size = train_labels.shape[0]  # [labels len, num of data points]
    return input_train, input_test, data_dim, train_labels, test_labels, label_size


# 2.1.1
def first_model_softmax_regression(selected_data, selected_labels, data_dim, label_size, num_of_hidden_layers=0,
                                   hidden_activation=None, loss="Cross Entropy", final_activation="softmax",
                                   learning_rate=0.1, grad_test=True, jac_test=False):
    basic_softmax_regression = NeuralNetwork(data_dim, label_size, num_of_hidden_layers, hidden_activation,
                                             loss, final_activation)
    selected_data, selected_labels = basic_softmax_regression.sgd_select_batch(selected_data, selected_labels)
    basic_softmax_regression.train(learning_rate, selected_data, selected_labels, grad_test, jac_test)


# 2.1.2
def second_model_least_squares(input_data, labels, data_dim, label_size, num_of_hidden_layers=0,
                               hidden_activation=None, loss="Least Squares",
                               final_activation="no final activation in Least Squares",
                               learning_rate=0.1, grad_test=False, jac_test=False):
    basic_least_squares = NeuralNetwork(data_dim, label_size, num_of_hidden_layers, hidden_activation, loss,
                                        final_activation)
    selected_data, selected_labels = basic_least_squares.sgd_select_batch(input_data, labels)
    basic_least_squares.train(learning_rate, selected_data, selected_labels, grad_test, jac_test)


# 2.1.3
def third_model_softmax_sgd(input_train, labels_train, input_test, labels_test, data_dim, label_size,
                            num_of_hidden_layers=0, hidden_activation=None, loss="Cross Entropy",
                            final_activation="softmax", learning_rate=0.1, grad_test=False, jac_test=False):
    sgd_softmax = NeuralNetwork(data_dim, label_size, num_of_hidden_layers, hidden_activation, loss, final_activation)
    loss_train_arr, success_train_arr, loss_test_arr, success_test_arr, index_arr = [], [], [], [], []
    success_percentage_arr = []
    success_sum = 0
    sgd_iter = 501
    for i in range(sgd_iter):
        selected_data, selected_labels = sgd_softmax.sgd_select_batch(input_train, labels_train)
        success_percentage_train, loss_train, y_pred = sgd_softmax.train(learning_rate, selected_data, selected_labels,
                                                                         grad_test, jac_test)
        sample_range = min(50, y_pred.shape[0])
        for j in range(sample_range):
            """
            y_pred = (batch_size, labels)
            selected_labels = (labels, batch_size)
            """
            subsample_train = y_pred[j]
            subsample_test = selected_labels[:, j]
            success = np.dot(subsample_train, subsample_test)
            success_sum += success
        success_percentage_arr.append(success_sum / sample_range)
        success_sum = 0
        success_percentage_test, loss_test = sgd_softmax.test(input_test, labels_test)
        loss_test_arr.append(loss_test)
        success_test_arr.append(success_percentage_test)
        loss_train_arr.append(loss_train)
        success_train_arr.append(success_percentage_train)
        index_arr.append(i)
    success_percentage_graph(success_percentage_arr, index_arr)
    plot_success_loss(index_arr,
                      {"binary success percentage train": success_train_arr, "loss train": loss_train_arr},
                      {"binary success percentage test": success_test_arr, "loss test": loss_test_arr})


# 2.2.1
def fourth_model_tanh_jac_one_layer(input_data, labels, data_dim, label_size, num_of_hidden_layers=1,
                                    hidden_activation="tanh", loss="Cross Entropy", final_activation="softmax",
                                    learning_rate=0.1, grad_test=False, jac_test=True):
    hidden_layer_model = NeuralNetwork(data_dim, label_size, num_of_hidden_layers, hidden_activation, loss,
                                       final_activation)
    hidden_layer_model.train(learning_rate, input_data, labels, grad_test, jac_test)


# 2.2.2
def fifth_model_res_net_tanh_jac(input_data, labels, data_dim, label_size, num_of_hidden_layers=1,
                                 hidden_activation="tanh", loss="Cross Entropy", final_activation="softmax",
                                 learning_rate=0.1, grad_test=False, jac_test=True, resnet=True):
    resnet_model = NeuralNetwork(data_dim, label_size, num_of_hidden_layers, hidden_activation, loss,
                                 final_activation, resnet)
    resnet_model.train(learning_rate, input_data, labels, grad_test, jac_test)


# 2.2.3
def sixth_model_l_layers(input_data, labels, data_dim, label_size, num_of_hidden_layers,
                         hidden_activation="tanh", loss="Cross Entropy", final_activation="softmax",
                         learning_rate=0.1, grad_test=True, jac_test=False):
    l_layers_model = NeuralNetwork(data_dim, label_size, num_of_hidden_layers, hidden_activation, loss,
                                   final_activation)
    selected_data, selected_labels = l_layers_model.sgd_select_batch(input_data, labels)
    l_layers_model.train(learning_rate, selected_data, selected_labels, grad_test, jac_test)

# 2.2.4
def



def main():
    folder_name = 'HW1_Data'
    file_names = [
        'GMMData.mat',  # input dims =[ , ], labels_dims =[ , ]
        'SwissRollData.mat',  # input dims =[ , ], labels_dims =[ , ]
        'PeaksData.mat'  # input dims =[ , ], labels_dims =[ , ]
    ]
    input_train, input_test, data_dim, labels_train, labels_test, label_size = extract_data(file_names[0], folder_name)
    #first_model_softmax_regression(input_train, labels_train, data_dim, label_size)
    #second_model_least_squares(input_train, labels_train, data_dim, label_size)
    #third_model_softmax_sgd(input_train, labels_train, input_test, labels_test, data_dim, label_size)
    #fourth_model_tanh_jac_one_layer(input_train[:, 0], labels_train[:, 0], data_dim, label_size)
    #fifth_model_res_net_tanh_jac(input_train[:, 0], labels_train[:, 0], data_dim, label_size)
    num_of_hidden_layers_L = 5
    sixth_model_l_layers(input_train, labels_train, data_dim, label_size, num_of_hidden_layers_L)
    L_arr = [0, 5, 10, 20]



if __name__ == '__main__':
    main()
