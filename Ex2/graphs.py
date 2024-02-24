import matplotlib.pyplot as plt
import numpy as np


def plot_signal_vs_time(data_examples, title, time='Time', signal='Signal Value'):
    # Shape data_examples [num_of_examples, sequence_length]
    sub = data_examples.shape[0]
    # Plotting
    fig, axes = plt.subplots(1, sub, figsize=(15, 5), sharey=True)
    time_steps = np.arange(50)
    for ax, example in zip(axes, data_examples):
        ax.plot(time_steps, example.detach().numpy())
        ax.set_xlabel(time)
        ax.set_ylabel(signal)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_grid_search(points):
    """
    :param points: each point -> [lr, grad_clip, hidden_units, best_loss]
    """
    points = np.array(points)
    x = points[:, 0]  # lr
    y = points[:, 1]  # grad_clip
    z = points[:, 2]  # hidden_units
    values = points[:, 3]  # best_loss

    # Normalize the values
    normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values))

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    scatter = ax.scatter(x, y, z, c=values, cmap='viridis', s=100)

    # Color bar indicating the values
    fig.colorbar(scatter, ax=ax, label='Loss')

    # Labels
    ax.set_xlabel('learning rate')
    ax.set_ylabel('gradient clipping')
    ax.set_zlabel('hidden units sizes')
    ax.set_title('3D Scatter Plot of Loss vs Different Params')

    # Show plot
    plt.show()


