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

    # print(images.shape)
    # Return images tensor with dimensions [num_of_images, num_of_rows, num_of_columns]
    return images, labels


def download_mnist_pixels():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: torch.flatten(x))])

    # Download and load the training data
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    # print(images.shape)   # [num_of_images, num_of_pixels]
    return images.unsqueeze(1).unsqueeze(3), labels


def split_train_test(data):
    test_size = data.shape[0] // 5
    # Generate random indices for validation set
    test_indices = torch.randperm(data.size(0))[:test_size]

    # Select rows for cross-validation based on random indices
    test = data[test_indices].unsqueeze(-1)  # Add a dimension for LSTM input

    # Select rows for training (exclude rows used for cross-validation)
    training_indices = torch.tensor([i for i in range(data.size(0)) if i not in test_indices])
    train = data[training_indices].unsqueeze(-1)  # Add a dimension for LSTM input

    return test, train


def download_snp500():
    # Assuming 'Symbol' is the column name for stock symbols
    file_path = 'data/SP500.csv'
    df = pd.read_csv(file_path)

    # Get unique symbols
    unique_symbols = df['symbol'].unique()

    # Convert to list (optional, since unique_symbols is already an array that can be iterated over)
    unique_symbols_list = list(unique_symbols)
    all_symbols = download_stock(unique_symbols_list)  # [num_of_stocks, sequence_length] = [505, 1007]
    # Check for None values
    clean = torch.tensor([[val is not None and not torch.isnan(val) for val in row] for row in all_symbols], dtype=torch.bool)
    # Filter out rows with None values
    filtered_stocks = all_symbols[clean.all(dim=1)]  # [num_clean_stocks, sequence_length] = [477, 1007]
    test, train = split_train_test(filtered_stocks)
    return test, train


def download_stock(stock_names):
    # Load the dataset
    file_path = 'data/SP500.csv'
    df = pd.read_csv(file_path)
    # In case one wants only one specific stock
    df = df[df['symbol'].isin(stock_names)]
    # Pivot the DataFrame so each company is a column, and each row is a date
    df_pivoted = df.pivot(index='date', columns='symbol', values='high')

    # Convert the clean DataFrame to a PyTorch tensor
    max_prices_tensor = torch.tensor(df_pivoted.values).float().t()

    # print(max_prices_tensor.shape)  [num_of_companies, sequence_length]
    return max_prices_tensor


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


