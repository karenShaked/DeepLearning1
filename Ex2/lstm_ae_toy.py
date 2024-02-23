import torch
import torch.nn as nn
import torch.optim as optim
from main import create_synthetic_data
from torch.utils.data import DataLoader
import numpy as np
from graphs import plot, plot_signal_vs_time
from lstm_ae_architecture import LSTMAutoencoder


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


def create_model(features_dim, hidden_units, sequence_len):
    """
    Create the LSTM Autoencoder model.
    """
    model = LSTMAutoencoder(features_dim, hidden_units, sequence_len)
    return model


def get_data_loaders(train_data, validation_or_test_data):
    shuffle = True
    train_loader = DataLoader(train_data, batch_size=256, shuffle=shuffle)
    validation_or_test_loader = DataLoader(validation_or_test_data, batch_size=256, shuffle=shuffle)
    return train_loader, validation_or_test_loader


def train_model(model, train_loader, criterion, optimizer):
    model.train()
    for batch in train_loader:
        # Forward pass
        outs = model(batch)
        loss = criterion(outs, batch)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_model(model, data_loader):
    model.eval()
    total_out = []
    total_in = []
    for batch in data_loader:
        with torch.no_grad():
            total_in.append(batch)
            outs = model(batch)  # (batch, sequence, features)
            total_out.append(outs)
    return torch.cat(total_in, dim=0), torch.cat(total_out, dim=0)


def evaluate_model(model, data_loader, criterion):
    avg_loss = None
    model.eval()
    for batch in data_loader:
        with torch.no_grad():
            outs = model(batch)
            loss = criterion(outs, batch)
            if avg_loss is None:
                avg_loss = loss.item()
            else:
                avg_loss = (loss.item() + avg_loss) / 2
    return avg_loss


def check_model(lr, grad_clip, hidden_units, train_data, validation_data, epochs):
    """
    Check the model's performance with given parameters
    """
    sequence_len = train_data.shape[1]
    features_dim = train_data.shape[2]

    # Model
    model = create_model(features_dim, hidden_units, sequence_len)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=grad_clip)

    # Data loader by batches
    train_loader, validation_loader = get_data_loaders(train_data, validation_data)

    min_loss, min_epoch = float('inf'), -1

    for epoch in range(epochs):
        train_model(model, train_loader, criterion, optimizer)
        avg_loss = evaluate_model(model, validation_loader, criterion)

        if avg_loss < min_loss:
            min_loss = avg_loss
            min_epoch = epoch

    return min_loss, min_epoch


def train_and_get_test_outputs(lr, grad_clip, hidden_units, train_data, test_data, epochs):
    # train_data Shape:  [num_of_data_point, sequence_len, features_dim] = [10000, 50, 1]
    # test_data Shape: [num_of_data_point, sequence_len, features_dim] = [2000, 50, 1]

    sequence_len = train_data.shape[1]
    features_dim = train_data.shape[2]

    # Model
    model = create_model(features_dim, hidden_units, sequence_len)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=grad_clip)

    # Data loader by batches
    train_loader, test_loader = get_data_loaders(train_data, test_data)
    for epoch in range(epochs):
        train_model(model, train_loader, criterion, optimizer)
    input_test, outs_test = test_model(model, test_loader)
    return input_test, outs_test


# Data
train, validation, test = create_synthetic_data()  # [num_of_data_point, sequence_len, features_dim]
examples = train[torch.randint(0, train.size(0), (3,))]
"""
# Select three examples from the synthetic dataset
examples = train[torch.randint(0, train.size(0), (3,))].squeeze(-1)  # [3, 50]

# 3.1.1 a graph of signal value vs. time for each  of data set
plot_signal_vs_time(examples, 'Signal Value vs. Time')

# 3.1.2 grid search
lr_array = [0.001, 0.01, 0.1]
grad_clip_array = [1.0, 5.0, 10.0]
hidden_units_array = [8, 16, 32]

loss, lr, gc, hidden_unit, iter_num = grid_search(lr_array, grad_clip_array, hidden_units_array, train, validation, 20)
print(f'AND THE WINNER IS:::::::::\nBest loss: {loss:.4f} at lr: {lr:.4f}, grad_clip: {gc:.4f},'
      f' hidden_units: {hidden_unit}')
"""
# original signal and its reconstruction
in_test, out_test = train_and_get_test_outputs(0.001, 1, 50, train, test, 20)
# lr, grad_clip, hidden_units, train_data, test_data, epochs
# Select three examples from the test and there outputs
examples_indexes = torch.randint(0, test.size(0), (3,))
examples_inputs = in_test[examples_indexes].squeeze(-1)   # [num_of_examples, sequence]
examples_outputs = out_test[examples_indexes].squeeze(-1)  # [num_of_examples, sequence]

for input_, output in zip(examples_inputs, examples_outputs):
    input_ = input_.unsqueeze(0)  # Shape: [1(feature), sequence]
    output = output.unsqueeze(0)  # Shape: [1(feature), sequence]
    in_out = torch.cat((input_, output), dim=0)  # Shape: [2, sequence]
    plot_signal_vs_time(in_out, 'Signal Value vs. Time \nInput vs.Output')
