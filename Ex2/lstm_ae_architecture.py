import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, in_features, hidden_units, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.lstm_encoder = nn.LSTM(in_features, hidden_units, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        _, (ht, _) = self.lstm_encoder(x)
        b = ht[self.num_layers - 1].view(-1, 1, self.hidden_units)
        a = ht[-1]
        return ht[self.num_layers - 1].view(-1, 1, self.hidden_units)


class Decoder(nn.Module):
    def __init__(self, hidden_units, num_layers, dropout, sequence, out_features):
        super(Decoder, self).__init__()
        self.sequence = sequence
        self.linear = nn.Linear(hidden_units, out_features)
        self.lstm_decoder = nn.LSTM(hidden_units, hidden_units, num_layers, batch_first=True, dropout=dropout)

    def forward(self, z):
        z = z.repeat(1, self.sequence, 1)
        output, (_, _) = self.lstm_decoder(z)
        return torch.tanh(self.linear(output))


class LSTMAutoencoder(nn.Module):
    def __init__(self, in_features, hidden_units, num_layers, dropout, sequence, out_features):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = Encoder(in_features, hidden_units, num_layers, dropout)
        self.decoder = Decoder(hidden_units, num_layers, dropout, sequence, out_features)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

