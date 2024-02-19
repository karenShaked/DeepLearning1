import torch
import torch.nn as nn
import torch.optim as optim
import random
from main import create_synthetic_data


# Define the LSTM Autoencoder
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sequence_length):
        super(LSTMAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length

        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # Decoder
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        # Encoder
        _, (hidden, _) = self.encoder(x)

        # Prepare decoder input
        # Use the hidden state as input to the decoder
        decoder_input = hidden.repeat(self.sequence_length, 1, 1).transpose(0, 1)

        # Decoder
        decoded_output, _ = self.decoder(decoder_input)
        return decoded_output


# Parameters
input_dim = 1  # Since we're treating each feature as a separate sequence
hidden_dim = 16  # Hidden layer size
sequence_length = 50  # Number of features in each data point

# Data
train, validation, test = create_synthetic_data()
train = train.unsqueeze(-1)  # Add a dimension for LSTM input

# Model
model = LSTMAutoencoder(input_dim, hidden_dim, sequence_length)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Reshape data for LSTM [Batch, Sequence Length, Features]
train = train.view(-1, sequence_length, input_dim)

# Training loop (simplified for illustration)
num_epochs = 5
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(train)
    loss = criterion(outputs, train)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
