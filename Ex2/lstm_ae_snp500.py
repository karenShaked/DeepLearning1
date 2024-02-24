import torch
from lstm_ae_model import LSTM_model, get_dims_copy_task
from create_data import download_snp500, download_stock
from graphs import prepare_plot_in_vs_out

HIDDEN = 500
NUM_LAYERS = 3
LR = 0.01
DROPOUT = 0.2
GRAD_CLIP = 1
EPOCHS = 100
BATCH = 32


def create_cross_validation(data):
    validation_size = data.shape[0] // 4
    # Generate random indices
    random_indices = torch.randperm(data.size(0))[:validation_size]
    # Select rows based on random indices
    cross_validation = data[random_indices]
    return cross_validation


def reconstruct_model():
    """
    3.3.2 Train the LSTM AE such that it reconstructs the stocks prices
    """
    model = LSTM_model(LR, in_features, HIDDEN, out_features, seq, NUM_LAYERS, DROPOUT, GRAD_CLIP)
    model.train(train, EPOCHS, BATCH, validation)
    out_test = model.reconstruct(test)
    prepare_plot_in_vs_out(test, out_test, sample_size=3)


def combine_data_for_predict(data):
    # Extract the first part (0-1005)
    before_xt = data[:, :1006, :]

    # Extract the second part (1-1006)
    after_yt = data[:, 1:, :]

    # Concatenate along the last dimension to create a tensor of shape (477, 1006, 2)
    before_after = torch.cat((before_xt, after_yt), dim=2)
    return before_after


def predict_model():
    pass


"""
# 3.3.1  graphs of the daily max value for the stocks AMZN and GOOGL
AMZN = download_stock(stock_names=["AMZN"])
plot_signal_vs_time(AMZN, "Amazon Stock Price vs. Time", time='Dates 1/2/2014 - 12/29/2017', signal='Stock Price')
GOOGL = download_stock(stock_names=["GOOGL"])
plot_signal_vs_time(GOOGL, "Google Stock Price vs. Time", time='Date  1/2/2014 - 12/29/2017', signal='Stock Price')
"""
test, train = download_snp500()  # [num_of_stocks, sequence_length, features] = [477, 1007, 1], test [95, 1007, 1], train [382, 1007, 1]
validation = create_cross_validation(train)  # [num_of_stocks, sequence_length, features] = [95, 1007, 1]
seq, in_features, out_features = get_dims_copy_task(test)

# 3.3.2
reconstruct_model()

# 3.3.3
predict_model()


