import matplotlib.pyplot as plt
import numpy as np

def generate_disc_data(n_samples, seed=27, plot=False):
    np.random.seed(27)

    X = np.random.uniform(low=0, high=1, size=(n_samples, 2))
    y = np.zeros(n_samples)
    center = np.array([0.5, 0.5])
    radius = 1 / (np.sqrt(2) * np.pi)
    distances = np.linalg.norm(X - center, axis=1)
    y[distances < radius] = 1

    if plot:
        plot_disc_data(X, y, center, radius)

    return X, y


def plot_disc_data(X, y, center, radius):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', label='Training Set')
    circle = plt.Circle(center, radius, color='gray', alpha=0.3)
    plt.gca().add_patch(circle)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Data Points')
    plt.legend()

    plt.show()
