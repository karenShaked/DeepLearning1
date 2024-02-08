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


def main():
    folder_path = 'HW1_Data'
    file_names = ['SwissRollData.mat', 'GMMData.mat', 'PeaksData.mat']

    for file_name in file_names:
        mat_data = scipy.io.loadmat(f'{folder_path}/{file_name}')
        train = 'Yt'
        labels_train = 'Ct'
        test = 'Yv'
        labels_test = 'Cv'
        input_train = mat_data[train]
        input_test = mat_data[test]
        data_dim = input_train.shape[0]  # [input features, num of data points]
        train_labels = mat_data[labels_train]
        test_labels = mat_data[labels_test]
        label_size = train_labels.shape[0]  # [num of labels, num of data points]
        num_of_hidden_layers = 0
        first_model_softmax_regression = NeuralNetwork(data_dim, label_size, num_of_hidden_layers, "ReLU",
                                                       "Cross Entropy", "softmax", True)
        learning_rate = 0.1
        grad_test = True
        jac_test = False
        loss_train_arr, success_train_arr, loss_test_arr, success_test_arr, index_arr = [], [], [], [], []
        success_percentage = []
        for i in range(1):
            selected_data, selected_labels = first_model_softmax_regression.sgd_select_batch(input_train, train_labels)
            success_percentage_train, loss_train, y_pred = first_model_softmax_regression.train(learning_rate,
                                                                                                selected_data,
                                                                                                selected_labels,
                                                                                                grad_test,
                                                                                                jac_test)
            if i % 4 == 0:
                success_sum = 0
                sample_range = min(50, y_pred.shape[0])
                for j in range(sample_range):
                    subsample_train = y_pred[j]
                    subsample_test = selected_labels[:, j]
                    success = np.dot(subsample_train, subsample_test)
                    success_sum += success
                success_percentage.append(success_sum / 50)
                print(f"train:\nloss {i} - {loss_train} \nsuccess percentage:{success_percentage_train}")
                success_percentage_test, loss_test = first_model_softmax_regression.test(input_test, test_labels)
                print(f"test:\nloss {i} - {loss_test} \nsuccess percentage:{success_percentage_test}\n\n")
                loss_test_arr.append(loss_test)
                success_test_arr.append(success_percentage_test)
                loss_train_arr.append(loss_train)
                success_train_arr.append(success_percentage_train)
                index_arr.append(i)
        success_percentage_graph(success_percentage, index_arr)
        plot_success_loss(index_arr,
                          {"binary success percentage train": success_train_arr, "loss train": loss_train_arr},
                          {"binary success percentage test": success_test_arr, "loss test": loss_test_arr})


if __name__ == '__main__':
    main()
