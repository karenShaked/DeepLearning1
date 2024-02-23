import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import random


def download_mnist_rows():
    transform = transforms.Compose([transforms.ToTensor()])

    # Download and load the training data
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    print(images.shape)
    # Return images tensor with dimensions [num_of_images, num_of_rows, num_of_columns]
    return images


def download_mnist_pixels():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: torch.flatten(x))])

    # Download and load the training data
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    print(images.shape)   # [num_of_images, num_of_pixels]
    return images


def download_snp500():
    # Load the dataset
    file_path = 'data/SP500.csv'
    df = pd.read_csv(file_path)
    # Pivot the DataFrame so each company is a column, and each row is a date
    df_pivoted = df.pivot(index='date', columns='symbol', values='close')

    # Convert the clean DataFrame to a PyTorch tensor
    close_prices_tensor = torch.tensor(df_pivoted.values).float().t()

    print(close_prices_tensor.shape)  # [num_of_companies, num_of_dates]
    return close_prices_tensor


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


if __name__ == '__main__':
    create_synthetic_data()

