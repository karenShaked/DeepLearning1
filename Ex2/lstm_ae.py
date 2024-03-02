from typing import Any
from torch import nn
import torch


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, device: Any):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.device = device
        self.h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size, requires_grad=True, device=self.device)
        self.c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size, requires_grad=True, device=self.device)

    def forward(self, input: torch.tensor) -> torch.tensor:
        output, (h_n, c_n) = self.lstm(input, (self.h_0, self.c_0))
        return torch.relu(h_n[-1]), output


class DecoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, input_seq_size, batch_size, device: Any):
        super(DecoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.device = device
        self.input_seq_size = input_seq_size
        self.h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size, requires_grad=True, device=self.device)
        self.c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size, requires_grad=True, device=self.device)

    def forward(self, z: torch.tensor) -> torch.tensor:
        z = z.unsqueeze(1)
        z = z.repeat(1, self.input_seq_size, 1)
        output, (h_n, c_n) = self.lstm(z, (self.h_0, self.c_0))
        return torch.relu(output)


class MnistAutoEncoder(nn.Module):
    def __init__(self, input_size, input_seq_size, hidden_size, num_layers, batch_size, decoder_output_size,
                 device: Any):
        super(MnistAutoEncoder, self).__init__()
        self.encoder = EncoderLSTM(input_size=input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   batch_size=batch_size,
                                   device=device)
        self.decoder = DecoderLSTM(input_size=hidden_size,
                                   hidden_size=decoder_output_size,
                                   num_layers=num_layers,
                                   input_seq_size=input_seq_size,
                                   batch_size=batch_size,
                                   device=device)
        self.func = nn.Linear(decoder_output_size * input_seq_size, input_size * input_seq_size)
        self.U_ht = nn.Linear(input_seq_size * hidden_size, 10)
        self.batch_size = batch_size
        self.input_size = input_size
        self.input_seq_size = input_seq_size

    def forward(self, x: torch.tensor) -> torch.tensor:
        z, enc_output = self.encoder(x)
        # decoder
        decoded = self.decoder(z)
        decoded = decoded.reshape(self.batch_size, -1)
        # prediction
        predictions = enc_output.reshape(self.batch_size, -1)
        predictions = torch.softmax(self.U_ht(predictions), dim=1)
        decoded = torch.relu(self.func(decoded))
        decoded = decoded.reshape(self.batch_size, self.input_seq_size, self.input_size)
        return predictions, decoded



