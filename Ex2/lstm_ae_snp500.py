import torch
from lstm_ae_model import LSTM_model, get_dims_copy_task
from create_data import download_snp500, download_stock
from graphs import prepare_plot_in_vs_out, plot_signal_vs_time, plot_group_arrays_one_graph, prepare_plot_in_vs_out_pred
from normalize_data import data_normalization
import numpy as np

HIDDEN = 50
NUM_LAYERS = 1
LR = 0.001
DROPOUT = 0
GRAD_CLIP = 1
EPOCHS = 5
BATCH = 128


def amazon_google_graphs():
    AMZN = download_stock(stock_names=["AMZN"])
    plot_signal_vs_time(AMZN, "Amazon Stock Price vs. Time", time='Dates 1/2/2014 - 12/29/2017', signal='Stock Price')
    GOOGL = download_stock(stock_names=["GOOGL"])
    plot_signal_vs_time(GOOGL, "Google Stock Price vs. Time", time='Date  1/2/2014 - 12/29/2017', signal='Stock Price')


def create_cross_validation(data):
    validation_size = data.shape[0] // 4
    # Generate random indices
    random_indices = torch.randperm(data.size(0))[:validation_size]
    # Select rows based on random indices
    cross_validation = data[random_indices]
    return cross_validation


def normalization():
    norm_train_ = data_normalization(train)
    norm_train_data_ = norm_train_.get_normalized_data()
    norm_validation_ = data_normalization(validation)
    norm_valid_data_ = norm_validation_.get_normalized_data()
    norm_test_ = data_normalization(test)
    norm_test_data_ = norm_test_.get_normalized_data()
    new_sequence = norm_test_.get_new_sequence()
    return norm_train_, norm_train_data_, norm_validation_, norm_valid_data_, norm_test_, norm_test_data_, new_sequence


def find_best_epoch(train_data, valid_data):
    print("Find Best Epoch")
    # Create and train model
    model = LSTM_model(LR, in_features, HIDDEN, out_features, seq, NUM_LAYERS, DROPOUT, GRAD_CLIP)
    train_loss, validation_loss = model.train(train_data, EPOCHS, BATCH, valid_data)

    # Plot the train and validation loss results
    arr_label = [[validation_loss, "validation_loss"]]
    plot_group_arrays_one_graph(arr_label, "Epochs vs. Loss validation", x_label='epochs', y_label='loss')

    # Find best epoch
    min_params_loss = np.argmin(validation_loss)
    best_epoch = min_params_loss
    print(f"best epoch detected: {best_epoch}")
    return best_epoch


def reconstruct_model():
    """
    3.3.2 Train the LSTM AE such that it reconstructs the stocks prices
    """
    print("Start Reconstruct Model 3.3.2")
    # Create, train and test the model
    recon_model = LSTM_model(LR, in_features, HIDDEN, out_features, seq, NUM_LAYERS, DROPOUT, GRAD_CLIP)
    recon_model.train(norm_train_data, EPOCHS, BATCH, norm_valid_data)
    out_test = recon_model.reconstruct(norm_test_data)

    # Denormalize the data
    out_test = norm_test.denormalize_data(out_test)

    # Plot some examples of the results
    prepare_plot_in_vs_out(test, out_test, sample_size=3)


def predict_model():
    """
        3.3.3 Train the LSTM AE such that it reconstructs the stocks prices and predicts one day in the future
    """
    print("Start Predict One Model 3.3.3")
    seq_pred = seq - 1
    pred_model = LSTM_model(LR, in_features, HIDDEN, out_features, seq_pred, NUM_LAYERS, DROPOUT, GRAD_CLIP, pred=True)
    train_loss, reconstruct_loss, predict_loss = pred_model.train_predict(norm_train_data, EPOCHS, BATCH)

    # Plot the train and validation loss results
    arr_label = [[train_loss, "train_loss"], [predict_loss, "predict_loss"]]
    plot_group_arrays_one_graph(arr_label, "training and prediction loss vs. time", x_label='epochs', y_label='loss')
    norm_test_data_pred, orig_last = norm_test.get_normalized_test_pred_one()
    out_test = pred_model.reconstruct_predict(norm_test_data_pred)

    # Denormalize the data
    out_test, predict_last = norm_test.denormalize_test_pred_one(out_test)
    distance = abs(predict_last - orig_last).squeeze(1)
    print(f"The mean error of predict denormalized: {distance.mean()}")
    prepare_plot_in_vs_out(test, out_test, sample_size=3)


def multi_step_predict_model():
    """
        3.3.4 Train the LSTM AE such that it predicts half of the sequence
    """
    print("Start Multi-Predict Model 3.3.4")
    seq_multi_pred = seq - 1
    orig_seq = test.shape[1]
    multi_pred_model = LSTM_model(LR, in_features, HIDDEN, out_features, seq_multi_pred, NUM_LAYERS, DROPOUT, GRAD_CLIP, pred=True)
    multi_pred_model.train_predict(norm_train_data, EPOCHS, BATCH)
    all_pred = None
    for sliding_window in range((orig_seq//2), orig_seq, 1):
        norm_test_data_pred = norm_test.get_normalized_test_pred_multi(sliding_window)
        out = multi_pred_model.reconstruct_predict(norm_test_data_pred)
        predict_one = out[:, -1:].unsqueeze(2)
        if all_pred is None:
            all_pred = predict_one
        else:
            all_pred = torch.cat((all_pred, predict_one), dim=1)
    orig_half_pred = torch.cat((test[:, :(orig_seq//2), :], all_pred), dim=1)
    orig_pred_denorm = norm_test.denormalize_data(orig_half_pred)
    prepare_plot_in_vs_out(test, orig_pred_denorm, sample_size=3)


# 3.3.1  graphs of the daily max value for the stocks AMZN and GOOGL
# amazon_google_graphs()

# Prepare data
test, train = download_snp500()  # [num_of_stocks, sequence_length, features] = [477, 1007, 1], test [95, 1007, 1], train [382, 1007, 1]
validation = create_cross_validation(train)  # [num_of_stocks, sequence_length, features] = [95, 1007, 1]

# Normalize the data to improve precision and performance
norm_train, norm_train_data, norm_validation, norm_valid_data, norm_test, norm_test_data, new_seq = normalization()
seq, in_features, out_features = get_dims_copy_task(norm_train_data)

# find best epoch
# best_epochs = find_best_epoch(norm_train_data, norm_valid_data)
# EPOCHS = best_epochs
# print(f"Found best epoch len {best_epochs}\n")

# 3.3.2
# reconstruct_model()

# 3.3.3
# predict_model()

# 3.3.4
multi_step_predict_model()


def predict_model_orig():
    """
        3.3.3 Train the LSTM AE such that it reconstructs the stocks prices and predicts one day in the future
    """
    print("Start Predict One Model 3.3.3")
    seq_pred = seq - 1
    pred_model = LSTM_model(LR, in_features, HIDDEN, out_features, seq_pred, NUM_LAYERS, DROPOUT, GRAD_CLIP, pred=True)
    train_loss, reconstruct_loss, predict_loss = pred_model.train_predict(norm_train_data, EPOCHS, BATCH)

    # Plot the train and validation loss results
    arr_label = [[train_loss, "train_loss"], [predict_loss, "predict_loss"]]
    plot_group_arrays_one_graph(arr_label, "training and prediction loss vs. time", x_label='epochs', y_label='loss')

    out_test = pred_model.reconstruct_predict(norm_test_data)

    # Denormalize the data
    out_test = norm_test.denormalize_data_predict(out_test, minus_pred=1)

    prepare_plot_in_vs_out_pred(test[:, :-new_seq, :], test[:, new_seq:, :], out_test, sample_size=3)
"""
    initial_vector = torch.randn(norm_test_data.shape[0], new_seq, norm_test_data.shape[2])
    all_pred = initial_vector
    first_round = True
    for sliding_window in range(seq_multi_pred):
        out_test = multi_pred_model.reconstruct_predict(norm_test_data[:, sliding_window:, :])
        out_test = norm_test.denormalize_data_predict(out_test, minus_pred=0, divide_pred=2)
        one_pred = out_test[:, -new_seq:, :]
        if first_round:
            first_round = False
            all_pred = one_pred
        else:
            all_pred = torch.cat((all_pred, one_pred), dim=1)

    input_before = test[:, 0:(seq_multi_pred * new_seq), :]
    true_pred = test[:, (seq_multi_pred * new_seq):(2 * seq_multi_pred * new_seq), :]
    prepare_plot_in_vs_out_pred(input_before, true_pred, all_pred, sample_size=3)
"""
