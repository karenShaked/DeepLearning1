import torch
from torch.utils.data import DataLoader
from lstm_ae_architecture import LSTMAutoencoder
from random import randint


def get_data_loader(data, batch, shuffle=True):
    data_loader = DataLoader(data, batch_size=batch, shuffle=shuffle)
    return data_loader


def get_dims_copy_task(data):
    # data Shape = [num_of_data_point, sequence_len, features_dim]
    seq = data.shape[1]
    input_features = data.shape[2]
    output_features = input_features  # copy task
    return seq, input_features, output_features


def data_for_predict(data):
    # Extract the first part
    before_xt = data[:, :-1, :]

    # Extract the second part
    after_yt = data[:, 1:, :]
    return before_xt, after_yt


class LSTM_model:
    def __init__(self, lr, in_features, hidden_units, out_features, sequence, num_layers, dropout, grad_clip=None, pred=False):
        self.grad_clip = grad_clip
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lstm = LSTMAutoencoder(in_features, hidden_units, num_layers, dropout, sequence, out_features, pred)
        optimizer = torch.optim.Adam
        self.optimizer = optimizer(self.lstm.parameters(), lr=lr)

    def train(self, train_data, epochs, batch_len, validation=None):
        # train_data [num_of_data_point, sequence_len, features_dim]
        # validation_data [num_of_data_point, sequence_len, features_dim]
        amount_data = train_data.shape[0]
        model = self.lstm.to(self.device)
        criterion = torch.nn.MSELoss().to(self.device)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
        # step_size - determines how often (in terms of epochs) to decrease the learning rate
        # gamma - determines how much to multiply the learning rate by after each step
        train_loss, validation_loss = [], []
        data_loader = get_data_loader(train_data, batch_len)
        for epoch in range(epochs):
            print(f"Starting epoch: {epoch+1}")
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
            train_loss.append(curr_loss / (amount_data // batch_len))  # avg of loss of all batches
            # In case we want to compute validation
            if validation is not None:
                v_data = validation.to(self.device)
                outputs = model(v_data)
                validation_loss.append(criterion(outputs, v_data).item())
        return train_loss, validation_loss

    def reconstruct(self, data):
        return self.lstm.to(self.device).forward(data.to(self.device))

    def train_predict(self, train_data, epochs, batch_len):
        model = self.lstm.to(self.device)
        criterion = torch.nn.MSELoss().to(self.device)

        total_loss = []
        reconstruct_loss = []
        predict_loss = []

        train_loader = get_data_loader(train_data, batch_len)

        for epoch in range(epochs):
            print(f'Starting epoch {epoch+1}')

            curr_total_loss = 0
            curr_reconstruct_loss = 0
            curr_predict_loss = 0
            for batch in train_loader:
                X, Y = data_for_predict(batch)
                X.to(self.device)
                Y.to(self.device)
                self.optimizer.zero_grad()

                # forward pass
                outputs, pred = model(X)
                loss_reconstruct = criterion(outputs, X)
                loss_predict = criterion(pred, Y.squeeze())

                loss = (loss_reconstruct + loss_predict) / 2

                # backward pass
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                self.optimizer.step()
                curr_total_loss += loss.item()
                curr_reconstruct_loss += loss_reconstruct.item()
                curr_predict_loss += loss_predict.item()

            divider = len(train_loader)
            total_loss.append(curr_total_loss / divider)
            reconstruct_loss.append(curr_reconstruct_loss / divider)
            predict_loss.append(curr_predict_loss / divider)

        return total_loss, reconstruct_loss, predict_loss

    def reconstruct_predict(self, data):
        data = data.to(self.device)
        X, Y = data_for_predict(data)
        data = X.to(self.device)
        reconstruct, predict = self.lstm.to(self.device).forward(data.to(self.device))
        return predict
