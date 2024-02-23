import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from graphs import plot, plot_signal_vs_time
from lstm_ae_architecture import LSTMAutoencoder

def create_synthetic_data():
    synthetic_tensor = torch.rand(10000, 50)
    for data in synthetic_tensor:
        random_int = random.randint(20, 30)
        data[random_int - 5: random_int + 5] *= 0.1
    synthetic_tensor = synthetic_tensor.unsqueeze(-1)  # Add a dimension for LSTM input
    print(synthetic_tensor.shape)  # [num_of_data_point, sequence_len, features_dim]
    train_data = synthetic_tensor[:6000]  # First 6000 for training
    validation_data = synthetic_tensor[6000:8000]  # Next 2000 for validation
    test_data = synthetic_tensor[8000:]  # Last 2000 for testing
    return train_data, validation_data, test_data


def create_model(features_dim, hidden_units, sequence_len):
    """
    Create the LSTM Autoencoder model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(features_dim, hidden_units, sequence_len, device)
    return model


def get_data_loaders(train_data, validation_or_test_data):
    shuffle = True
    train_loader = DataLoader(train_data, batch_size=256, shuffle=shuffle)
    validation_or_test_loader = DataLoader(validation_or_test_data, batch_size=256, shuffle=shuffle)
    return train_loader, validation_or_test_loader


def train_model(model, train_loader, criterion, optimizer, grad_clip=None):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outs = model(batch)
        loss = criterion(outs, batch)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
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
# original signal and its reconstruction
in_test, out_test = train_and_get_test_outputs(0.001, 1, 40, train, test, 20)
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

