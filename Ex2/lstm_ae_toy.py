import torch
from create_data import create_synthetic_data
import numpy as np
from graphs import plot_grid_search, plot_signal_vs_time
from lstm_ae_model import LSTM_model

NUM_LAYERS = 1
DROPOUT = 0
EPOCHS = 100
BATCH = 128


def get_dims_copy_task(data):
    # data Shape = [num_of_data_point, sequence_len, features_dim]
    seq = data.shape[1]
    input_features = data.shape[2]
    output_features = input_features  # copy task
    return seq, input_features, output_features


def grid_search(lr_arr, grad_clip_arr, hidden_units_arr, train_data, validation_data, epochs, seq, input_features,
                output_features):
    # train_data Shape = [num_of_data_point, sequence_len, features_dim] = [10000, 50, 1]
    # validation_data Shape = [num_of_data_point, sequence_len, features_dim] = [2000, 50, 1]
    min_loss, min_lr, min_gc, min_hidden, min_iter = float('inf'), -1, -1, -1, -1
    points = []
    for lr in lr_arr:
        for grad_clip in grad_clip_arr:
            for hidden_units in hidden_units_arr:
                model = LSTM_model(lr, input_features, hidden_units, output_features, seq, num_layers=NUM_LAYERS,
                                   dropout=DROPOUT, grad_clip=grad_clip)
                train_loss, validation_loss = model.train(train_data, EPOCHS, BATCH, validation_data)
                min_params_loss = min(validation_loss)
                if min_loss > min_params_loss:
                    min_loss, min_lr, min_gc, min_hidden, min_iter = min_params_loss, lr, grad_clip, hidden_units, np.argmin(validation_loss)
                # Save the points for plot
                point = np.array([lr, grad_clip, hidden_units, min_params_loss])
                points.append(point)

    # plot the grid search:
    plot_grid_search(points)
    return min_loss, min_lr, min_gc, min_hidden, min_iter


def prepare_plot_in_vs_out(in_test, out_test, sample_size):
    examples_indexes = torch.randint(0, test.size(0), (sample_size,))
    examples_inputs = in_test[examples_indexes].squeeze(-1)  # [num_of_examples, sequence]
    examples_outputs = out_test[examples_indexes].squeeze(-1)  # [num_of_examples, sequence]

    for input_, output_ in zip(examples_inputs, examples_outputs):
        input_ = input_.unsqueeze(0)  # Shape: [1(feature), sequence]
        output_ = output_.unsqueeze(0)  # Shape: [1(feature), sequence]
        in_out = torch.cat((input_, output_), dim=0)  # Shape: [2, sequence]
        plot_signal_vs_time(in_out, 'Signal Value vs. Time \nInput vs.Output')


# Data
train, validation, test = create_synthetic_data()  # [num_of_data_point, sequence_len, features_dim]

# Select three examples from the synthetic dataset
examples = train[torch.randint(0, train.size(0), (3,))].squeeze(-1)  # [3, 50]

# 3.1.1 a graph of signal value vs. time for each  of data set
plot_signal_vs_time(examples, 'Signal Value vs. Time')

# 3.1.2 grid search
lr_array = [0.001, 0.01, 0.1]
grad_clip_array = [1.0, 5.0, 10.0]
hidden_units_array = [8, 16, 32]

sequence, in_features, out_features = get_dims_copy_task(test)

loss, lr, gc, hidden_unit, epochs = grid_search(lr_array, grad_clip_array, hidden_units_array, train, validation, epochs=EPOCHS,
                                                seq=sequence, input_features=in_features, output_features=out_features)

print(f'THE BEST MODEL PARAMS ARE :::::::::\n'
      f'Best loss: {loss:.4f} at lr: {lr:.4f}, grad_clip: {gc:.4f}, hidden_units: {hidden_unit}')


# plot input vs. reconstruct
best_model = LSTM_model(lr, in_features, hidden_unit, out_features, sequence, num_layers=NUM_LAYERS, dropout=DROPOUT, grad_clip=gc)
best_model.train(train, epochs=epochs, batch=BATCH, validation=validation)
output = best_model.reconstruct(test)
prepare_plot_in_vs_out(test, output, sample_size=3)
