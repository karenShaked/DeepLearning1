from matplotlib import pyplot as plt, transforms
from scipy import datasets
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from lstm_ae import MnistAutoEncoder
from create_data import download_mnist_pixels, download_mnist_rows


class lstm_ae_mnist:
    def __init__(self, images, labels, classification=False, pixels=False):
        self.batch_size = images.shape[0]
        self.num_of_rows = images.shape[2]
        self.num_of_columns = images.shape[3]
        self.images = images.view(self.batch_size, self.num_of_rows, self.num_of_columns)
        self.labels = labels
        self.output_images = self.images
        self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MnistAutoEncoder(input_size=self.num_of_columns, input_seq_size=self.num_of_rows, hidden_size=64,
                                      num_layers=2, batch_size=self.batch_size, decoder_output_size=self.num_of_rows,
                                      device=self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # Define optimizer
        self.criterion = nn.MSELoss()  # Define loss function
        self.classification = classification
        self.pixels = pixels

    def train(self, num_epochs):
        self.model.train()  # Set the model to training mode
        running_loss = 0.0
        self.optimizer.zero_grad()  # Zero the parameter gradients
        loss_arr = []
        prec_arr = []
        for epoch in range(num_epochs):
            predictions, outputs = self.model.forward(self.images)  # Forward pass
            if self.classification:
                mse_loss, ce_loss = self.mse_ce_loss(outputs, self.images, predictions, self.labels)
                loss = mse_loss + ce_loss
            else:
                loss = self.criterion(outputs, self.images)  # Compute the loss
            loss.backward()  # Backward pass
            self.optimizer.step()  # Optimize
            running_loss += loss.item()
            self.output_images = outputs
            loss_arr.append(loss.item() / len(self.images))
            if self.classification:
                prec_arr.append(self.prediction(predictions))
        self.print_mnist()
        self.plot(num_epochs, loss_arr, prec_arr)
        self.test_data()

    def mse_ce_loss(self, data_rec, data, out_labels, labels):
        mse_loss = self.criterion(data_rec, data)
        additional_critetrion = nn.CrossEntropyLoss()
        ce_loss = additional_critetrion(out_labels, labels)

        return mse_loss, ce_loss

    def print_mnist(self):
        images = self.images
        output_images = self.output_images
        if self.pixels:
            images = images.reshape(-1, 28, 28)
            output_images = output_images.reshape(-1, 28, 28)
        images_np = images.numpy()
        output_images_np = output_images.detach().numpy()

        for i in range(5):
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            temp = images_np[i].squeeze()
            axes[0].imshow(temp, cmap='gray')
            axes[0].set_title('Original')
            axes[0].axis('off')

            axes[1].imshow(output_images_np[i].squeeze(), cmap='gray')
            axes[1].set_title('Reconstructed')
            axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    def plot(self, num_epochs, loss_arr, prec_arr):
        plt.plot(range(num_epochs), loss_arr)
        plt.xlabel('num of epochs')
        plt.ylabel('loss')
        plt.title('loss vs. epochs')
        plt.show()

        if self.classification:
            plt.plot(range(num_epochs), prec_arr)
            plt.xlabel('num of epochs')
            plt.ylabel('accuracy')
            plt.title('accuracy vs. epochs')
            plt.show()

    def test_data(self):
        if self.pixels:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Lambda(lambda x: torch.flatten(x))])
        else:
            transform = transforms.Compose([transforms.ToTensor()])

        # Download and load the training data
        testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

        dataiter = iter(testloader)
        test_images, test_labels = next(dataiter)
        if self.pixels:
            test_images = test_images.unsqueeze(1).unsqueeze(3)
        test_images = test_images.view(test_images.shape[0], test_images.shape[2], test_images.shape[3])

        predictions, test_outputs = self.model.forward(test_images)  # Forward pass

        correct = 0
        total = 0

        with torch.no_grad():
            _, predicted = torch.max(predictions.data, 1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()
            print(f"test accuracy: {100 * correct / total}")

    def prediction(self, predictions):
        correct = 0
        total = 0

        with torch.no_grad():
            _, predicted = torch.max(predictions.data, 1)
            total += self.labels.size(0)
            correct += (predicted == self.labels).sum().item()
            return 100 * correct / total


# 3.2.2
images, labels = download_mnist_rows()
LSTM = lstm_ae_mnist(images, labels, classification=True)  
LSTM.train(1000)


# 3.2.3
images, labels = download_mnist_pixels()
LSTM = lstm_ae_mnist(images, labels, classification=True, pixels=True)  # pixels=True
LSTM.train(10)

