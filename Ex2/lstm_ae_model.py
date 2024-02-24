import torch
from torch.utils.data import DataLoader
from lstm_ae_architecture import LSTMAutoencoder


def get_data_loader(data, batch, shuffle=True):
    data_loader = DataLoader(data, batch_size=batch, shuffle=shuffle)
    return data_loader


class LSTM_model:
    def __init__(self, lr, in_features, hidden_units, out_features, sequence, num_layers, dropout, grad_clip=None):
        self.grad_clip = grad_clip
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lstm = LSTMAutoencoder(in_features, hidden_units, num_layers, dropout, sequence, out_features)
        optimizer = torch.optim.Adam
        self.optimizer = optimizer(self.lstm.parameters(), lr=lr)

    def train(self, train_data, epochs, batch, validation=None):
        # train_data [num_of_data_point, sequence_len, features_dim] = [10000, 50, 1]
        # validation_data [num_of_data_point, sequence_len, features_dim] = [2000, 50, 1]
        amount_data = train_data.shape[0]
        model = self.lstm.to(self.device)
        criterion = torch.nn.MSELoss().to(self.device)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
        # step_size - determines how often (in terms of epochs) to decrease the learning rate
        # gamma - determines how much to multiply the learning rate by after each step
        train_loss, validation_loss = [], []
        data_loader = get_data_loader(train_data, batch)
        for epoch in range(epochs):
            curr_loss = 0  # initialize this epoch loss
            for batch in data_loader:
                # forward
                self.optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, batch)
                # backpropagation
                loss.backward()  # Computes gradients of the loss with respect to all model parameters
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                self.optimizer.step()  # Updates model parameters based on the computed gradients
                curr_loss += loss.item()
            lr_scheduler.step()
            train_loss.append(curr_loss / (amount_data // batch))  # avg of loss of all batches
            # In case we want to compute validation
            if validation is not None:
                v_data = validation.to(self.device)
                outputs = model(v_data)
                validation_loss.append(criterion(outputs, v_data).item())
        return train_loss, validation_loss

    def reconstruct(self, data):
        return self.lstm.to(self.device).forward(data.to(self.device))


