import torch.nn as nn
import torch
"""
class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # , dropout=0.2

    def forward(self, input):
        _, (h_n, _) = self.lstm(input)
        return h_n[-1]  # Return the last layer's hidden state


class DecoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, input_seq_size):
        super(DecoderLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # , dropout=0.2
        self.input_seq_size = input_seq_size

    def forward(self, z):
        batch_size = z.size(0)
        z = z.unsqueeze(1).repeat(1, self.input_seq_size, 1)  # Prepare z for the sequence generation
        output, _ = self.lstm(z)
        return output


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size,  input_seq_size, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = EncoderLSTM(input_size, hidden_size, num_layers)
        self.decoder = DecoderLSTM(hidden_size, input_size, num_layers, input_seq_size)
        self.fc = nn.Linear(input_size, input_size)

    def forward(self, x):
        z = self.encoder(x)
        decoded = self.decoder(z)
        reconstruct = self.fc(decoded)
        return reconstruct

"""


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.device = device

    def forward(self, input):
        batch_size = input.size(0)
        h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size, requires_grad=True, device=self.device)
        c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size, requires_grad=True, device=self.device)
        out, (h_n, c_n) = self.lstm(input, (h_0, c_0))
        return torch.relu(h_n[-1])  # Return the last layer's hidden state without activation.


class DecoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, input_seq_size, device):
        super(DecoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.device = device
        self.input_seq_size = input_seq_size

    def forward(self, z):
        batch_size = z.size(0)
        z = z.unsqueeze(1).repeat(1, self.input_seq_size, 1)  # Prepare for decoding
        h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size, requires_grad=True, device=self.device)
        c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size, requires_grad=True, device=self.device)
        output, (h_n, c_n) = self.lstm(z, (h_0, c_0))
        return torch.relu(output)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, input_seq_size, device, num_layers=3):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = EncoderLSTM(input_size, hidden_size, num_layers, device)
        self.decoder = DecoderLSTM(hidden_size, input_size, num_layers, input_seq_size, device)
        self.fc = nn.Linear(input_size, input_size)

    def forward(self, x):
        z = self.encoder(x)
        decoded = self.decoder(z)
        reconstruct = self.fc(decoded)
        return torch.relu(reconstruct)


