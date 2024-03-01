import numpy as np
import torch


def find_closest_factors(n):
    closest_pair = (1, n)
    min_diff = n - 1
    for i in range(int(np.sqrt(n))):
        if n % (i + 1) == 0:
            factor1, factor2 = (i + 1), n // (i + 1)
            diff = abs(factor2 - factor1)
            if diff < min_diff:
                min_diff = diff
                closest_pair = (factor1, factor2)
    return closest_pair


def reshape_seq_to_closest_factors(tensor):
    # Reshape the huge sequence to smaller dimensions:
    # 1) Improve performance
    # 2) Sharpen the learning of the lstm

    batch_size = tensor.shape[0]
    orig_seq = tensor.shape[1]
    closest_factors = find_closest_factors(orig_seq)
    new_shape = (batch_size,) + closest_factors
    return tensor.view(new_shape), new_shape


class data_normalization:
    def __init__(self, data):
        self.orig_data = data
        self.orig_shape = data.shape
        self.reshaped_data, (self.batch, self.new_sequence, self.new_features) = reshape_seq_to_closest_factors(data)
        print(f"{self.batch, self.new_sequence, self.new_features}")
        self.normalized_data, self.mean, self.std = self.normalize_per_sequence()
        self.normalized_data = self.normalized_data.view(self.batch * self.new_sequence, self.new_features, 1)
        self.norm_shape_orig = self.normalized_data.view(self.orig_shape)

    def get_calculation_shape(self):
        return self.batch, self.new_sequence, self.new_features

    def get_new_sequence(self):
        return self.new_sequence

    def get_normalized_data(self):
        return self.normalized_data

    def normalize_per_sequence(self):
        # Calculate the mean and std per sub-sequence for each batch
        # mean and std have shapes of (batch, num_of_subsequence, 1) since we are calculating them
        # across the last dimension (features_in_each_subsequence)

        self.mean = torch.mean(self.reshaped_data, dim=2, keepdim=True)
        self.std = torch.std(self.reshaped_data, dim=2, keepdim=True)

        # Avoid division by zero by setting std to 1 where it's 0
        self.std = torch.where(self.std == 0, torch.ones_like(self.std), self.std)

        # Normalize data
        self.normalized_data = (self.reshaped_data - self.mean) / self.std

        return self.normalized_data, self.mean, self.std

    def get_normalized_test_pred_one(self):
        norm_shape_orig_last_seq = self.norm_shape_orig[:, -self.new_features:, :]
        last_element = self.orig_data[:, -1:, :].squeeze(2)
        return norm_shape_orig_last_seq, last_element

    def get_normalized_test_pred_multi(self, sliding_window):
        norm_shape_orig_i_seq = self.norm_shape_orig[:, sliding_window: (sliding_window+self.new_features), :]
        return norm_shape_orig_i_seq

    def denormalize_test_pred_one(self, data):
        data = data.view(self.batch, 1, (self.new_features - 1))
        predict_last = data[:, :, -1]
        orig_seq = self.orig_shape[1]
        for i in range(self.batch):
            self.norm_shape_orig[i, orig_seq-1, 0] = predict_last[i, 0]
        data = self.norm_shape_orig.view(self.batch, self.new_sequence, self.new_features)
        denormalized_data_reshaped = (data * self.std) + self.mean
        denormalized_test = denormalized_data_reshaped.view(self.batch, self.new_sequence * self.new_features, 1)
        predict_last_denormalized = denormalized_test[:, -1:, :].squeeze(2)
        return denormalized_test, predict_last_denormalized

    def denormalize_data(self, data):
        # reshape data to subsequences
        data = data.view(self.batch, self.new_sequence, self.new_features)
        # denormalize data
        denormalized_data_reshaped = (data * self.std) + self.mean
        # reshape data to original sequence
        denormalized_data = denormalized_data_reshaped.view(self.batch, self.new_sequence * self.new_features, 1)
        return denormalized_data

    def denormalize_data_predict(self, data, minus_pred=0, divide_pred=1):
        # reshape data to subsequences
        data = data.view(self.batch, self.new_sequence, (self.new_features-minus_pred)//divide_pred)
        # denormalize data
        denormalized_data_reshaped = (data * self.std) + self.mean
        # reshape data to original sequence
        denormalized_data = denormalized_data_reshaped.view(self.batch, self.new_sequence * ((self.new_features-minus_pred)//divide_pred), 1)
        return denormalized_data


def test_normalization():
    batch_size, sequence_length, features = 2, 4, 1  # Example dimensions
    synthetic_data = torch.rand(batch_size, sequence_length, features) * 10  # Random data scaled by 10
    normalizer = data_normalization(synthetic_data)
    normalized_data = normalizer.get_normalized_data()
    denormalized_data = normalizer.denormalize_data(normalized_data)
    print(f"{normalizer.get_calculation_shape()}")
    print(f"Original: {synthetic_data}")
    print(f"Normalized: {normalized_data}")
    print(f"Denormalized: {denormalized_data}")


# test_normalization()

