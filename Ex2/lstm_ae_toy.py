import torch
import torch.nn as nn
import torch.optim as optim
from main import create_synthetic_data
from torch.utils.data import DataLoader
import numpy as np
from graphs import plot, plot_signal_vs_time


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sequence_length):
        super(LSTMAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length

        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        # Decoder
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):  # x dims [batch_size, sequence_len, features]
        # Encoder
        _, (hidden, _) = self.encoder(x)

        # Prepare decoder input
        # Use the hidden state as input to the decoder
        decoder_input = hidden.repeat(self.sequence_length, 1, 1).transpose(0, 1)

        # Decoder
        decoded_output, _ = self.decoder(decoder_input)
        return decoded_output


def grid_search(lr_array, grad_clip_array, hidden_units_array, train_data, validation_data, epochs):
    min_loss, min_lr, min_gc, min_hidden, min_iter = float('inf'), -1, -1, -1, -1
    points = []
    for lr in lr_array:
        for grad_clip in grad_clip_array:
            for hidden_units in hidden_units_array:
                best_loss, best_iter = check_model(lr, grad_clip, hidden_units, train_data, validation_data, epochs)

                # Save for plot the points
                point = np.array([lr, grad_clip, hidden_units, best_loss])
                points.append(point)

                print(f'Best loss: {best_loss:.4f} at lr: {lr:.4f}, grad_clip: {grad_clip:.4f}, '
                      f'hidden_units: {hidden_units}')
                print("================================")
                if min_loss > best_loss:
                    min_loss, min_lr, min_gc, min_hidden, min_iter = best_loss, lr, grad_clip, hidden_units, best_iter
    # plot the grid search:
    plot(points)
    return min_loss, min_lr, min_gc, min_hidden, min_iter


def check_model(lr, grad_clip, hidden_units, train_data, validation_data, epochs):
    """
    :param train_data: [num_of_data_point, sequence_len, features_dim]
    :param validation_data: [num_of_data_point, sequence_len, features_dim]
    :return:
    """
    sequence_len = train_data.shape[1]
    features_dim = train_data.shape[2]   # features_dim specifies the dimension of each feature in the input sequence

    # Model
    model = LSTMAutoencoder(features_dim, hidden_units, sequence_len)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=grad_clip)

    # Data loader by batches
    train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=512, shuffle=True)

    min_loss, min_epoch = float('inf'), -1

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            # Forward pass
            outs = model(batch)
            loss = criterion(outs, train_data)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluation of validation set
        avg_loss = None
        model.eval()
        for batch in validation_loader:
            with torch.no_grad():
                outs = model(batch)
                loss = criterion(outs, validation_data)
                if not avg_loss:
                    avg_loss = loss.item()
                else:
                    avg_loss = (loss.item() + avg_loss) / 2
        if avg_loss < min_loss:
            min_loss = avg_loss
            min_epoch = epoch

    return min_loss, min_epoch


# Data
train, validation, test = create_synthetic_data()  # [num_of_data_point, sequence_len, features_dim]

# Select three examples from the synthetic dataset
examples = train[torch.randint(0, train.size(0), (3,))].squeeze(-1)

# 3.1 a graph of signal value vs. time for each  of data set
plot_signal_vs_time(examples)

# 3.2 grid search
lr_array = [0.001, 0.01, 0.1]
grad_clip_array = [1.0, 5.0, 10.0]
hidden_units_array = [8, 16, 32]

"""
loss, lr, gc, hidden_unit, iter_num = grid_search(lr_array, grad_clip_array, hidden_units_array, train, validation, 20)
print(f'AND THE WINNER IS:::::::::\nBest loss: {loss:.4f} at lr: {lr:.4f}, grad_clip: {gc:.4f},'
      f' hidden_units: {hidden_unit}')
"""