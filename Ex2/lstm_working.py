"""import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from graphs import plot_signal_vs_time
from lstm_ae_architecture import LSTMAutoencoder as AE

def create_synthetic_data():
    synthetic_tensor = torch.rand(10000, 50, 1)
    for data in synthetic_tensor:
        random_int = random.randint(20, 30)
        data[random_int - 5: random_int + 6] *= 0.1
    print(synthetic_tensor.shape)  # [num_of_data_point, sequence_len, features_dim]
    train_data = synthetic_tensor[:6000]  # First 6000 for training
    validation_data = synthetic_tensor[6000:8000]  # Next 2000 for validation
    test_data = synthetic_tensor[8000:]  # Last 2000 for testing
    return train_data, validation_data, test_data


def create_model(features_dim, hidden_units, sequence_len):

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


class AE_TOY:
    def __init__(self):
        self.train_data = train
        self.validation_data = validation
        self.test_data = test
        self.lr = 0.001
        self.batch = 100
        self.epochs = 200
        self.hidden_state_sz = 30
        self.input_sz = 1
        self.seq_sz = 50
        self.output_sz = 1
        self.grad_clip = 1
        self.num_layers = 1
        self.dropout = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.auto_encoder = AE(self.input_sz, self.hidden_state_sz, self.num_layers, self.dropout, self.seq_sz, self.output_sz)
        optimizer = torch.optim.Adam
        self.optimizer = optimizer(self.auto_encoder.parameters(), lr=self.lr)

    def train(self):
        amount_data = self.train_data.size(dim=0)
        model = self.auto_encoder.to(self.device)
        criterion = torch.nn.MSELoss().to(self.device)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 100, 0.5)

        train_loss = []
        validation_loss = []

        for epoch in range(self.epochs):
            rnd_ind = np.random.permutation(amount_data)

            curr_loss = 0

            for b in range(math.floor(amount_data / self.batch)):
                ind = rnd_ind[b * self.batch: (b + 1) * self.batch]
                train_ind = self.train_data[ind, :, :].to(self.device)
                self.optimizer.zero_grad()

                # forward pass
                outputs = model(train_ind)
                loss = criterion(outputs, train_ind)

                # backward pass
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                self.optimizer.step()
                curr_loss += loss.item()

            lr_scheduler.step()
            train_loss.append(curr_loss / math.floor(amount_data / self.batch))

            v_data = self.validation_data.to(self.device)
            outputs = model(v_data)
            validation_loss.append(criterion(outputs, v_data).item())

        return train_loss, validation_loss

    def reconstruct(self, data):
        return self.auto_encoder.to(self.device).forward(data.to(self.device))


# Data
train, validation, test = create_synthetic_data()  # [num_of_data_point, sequence_len, features_dim]
#lstm = AE_TOY()
#lstm.train()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
auto_encoder = AE(1, 30, 1, 0, 50, 1)
optimizer = torch.optim.Adam
optimizer = optimizer(auto_encoder.parameters(), lr=0.001)
amount_data = train.size(dim=0)
model = auto_encoder.to(device)
criterion = torch.nn.MSELoss().to(device)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 0.5)

train_loss = []
validation_loss = []
for epoch in range(100):
    rnd_ind = np.random.permutation(amount_data)
    grad_clip = 1
    curr_loss = 0
    batch = 100
    for b in range(math.floor(amount_data / batch)):
        ind = rnd_ind[b * batch: (b + 1) * batch]
        train_ind = train[ind, :, :].to(device)
        optimizer.zero_grad()

        # forward pass
        outputs = model(train_ind)
        loss = criterion(outputs, train_ind)

        # backward pass
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        curr_loss += loss.item()

    lr_scheduler.step()
    train_loss.append(curr_loss / math.floor(amount_data / batch))

    v_data = validation.to(device)
    outputs = model(v_data)
    validation_loss.append(criterion(outputs, v_data).item())
#out = auto_encoder.reconstruct(test)
out = auto_encoder.to(device).forward(test.to(device))
examples_indexes = torch.randint(0, test.size(0), (3,))
examples_inputs = test[examples_indexes].squeeze(-1)   # [num_of_examples, sequence]
examples_outputs = out[examples_indexes].squeeze(-1)  # [num_of_examples, sequence]

for input_, output in zip(examples_inputs, examples_outputs):
    input_ = input_.unsqueeze(0)  # Shape: [1(feature), sequence]
    output = output.unsqueeze(0)  # Shape: [1(feature), sequence]
    in_out = torch.cat((input_, output), dim=0)  # Shape: [2, sequence]
    plot_signal_vs_time(in_out, 'Signal Value vs. Time \nInput vs.Output')

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
    """
"""
def create_model(features_dim, hidden_units, sequence_len):
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
    
    Check the model's performance with given parameters
    
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
    return input_test, outs_test"""


