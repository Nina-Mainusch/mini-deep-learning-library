# You must implement a test executable named test.py that imports your framework and
# • Generates a training and a test set of 1, 000 points sampled uniformly in [0, 1]2
# , each with a
# label 0 if outside the disk centered at (0.5, 0.5) of radius 1/√2π, and 1 inside,
# • builds a network with two input units, one output unit, three hidden layers of 25 units,
# • trains it with MSE, logging the loss,
# • computes and prints the final train and the test errors

import torch
import torch.nn as nn
import unittest
from matplotlib import pyplot as plt
from torch import optim

import numpy as np
from activationfunctions import ReLU, Sigmoid
from data import generate_disc_data
# from sklearn.model_selection import train_test_split
from linearlayer import LinearLayer
from lossfunctions import MSE
from optimizer import SGD

from sequential import Sequential


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(2, 25)
        self.linear2 = nn.Linear(25, 25)
        self.linear3 = nn.Linear(25, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, a):
        # a = image.view(-1, 28*28)
        a = self.relu(self.linear1(a))
        a = self.relu(self.linear2(a))
        a = self.sigmoid(self.linear3(a))
        # a = self.final(a)
        return a

class TestUniformDisk(unittest.TestCase):
    def __init__(self, methodName: str = "runEndToEndTest") -> None:
        super().__init__(methodName)

        self.network = Sequential(
            [LinearLayer(2, 25), ReLU(), LinearLayer(25, 25), ReLU(), LinearLayer(25, 1), ReLu()])  # , LinearLayer(2, 2), LinearLayer(2, 1)

        self.torch_network = MLP()
        self.torch_optimizer = optim.SGD(self.torch_network.parameters(), lr=0.001)
        self.torch_loss_function = nn.MSELoss()

        self.optimizer = SGD(lr=0.001)
        self.loss_function = MSE()
        self.training_samples = 1000
        self.X, self.y = generate_disc_data(self.training_samples)

        self.epochs = 50

        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        #     X, y, test_size=0.2, random_state=42)

    def test_trains_network_with_MSE(self):
        """
        """
        # print(" Initial weights ", self.network.get_layers()
        #       [0].get_parameters())

        for epoch in range(self.epochs):
            average_loss = 0
            for datapoint, target in zip(self.X, self.y):
                output = self.network.forward(datapoint)
                loss = self.loss_function(output, target)
                mse_derivative = self.loss_function.backward(output, target)
                self.network.backward(mse_derivative)
                self.optimizer.step(self.network)
                average_loss += loss
            if epoch % 10 == 0:
                print(">> Epoch: ", epoch, "Avg Loss: ", round(
                    average_loss/self.training_samples, 4))

        plot_decision_boundary(self.network, self.X, self.y)


        # TODO: problem: something goes wrong. Just remembered, sigmoid and MSE might need something special? Derivative?
        # Think about this. Loss is not decreasing


        # print("")
        # print("--- HOMEBREW")
        # print(" Final weights ", self.network.get_layers()[0].get_parameters())
        # print("")


        # Training with PyTorch
        # print(" Initial weights ", self.torch_network.linear1.weight)
        for epoch in range(self.epochs):
            average_loss = 0
            for datapoint, target in zip(self.X, self.y):
                self.torch_optimizer.zero_grad()
                output = self.torch_network(torch.tensor(datapoint).float())
                loss = self.torch_loss_function(output, torch.tensor(target).float())
                loss.backward()
                self.torch_optimizer.step()
                average_loss += loss
            if epoch % 10 == 0:
                print(">> TORCH >> Epoch: ", epoch, "Avg Loss: ", round(
                    average_loss.item()/self.training_samples, 4))
        # print("")
        # print("--- TORCH")
        # print(" Final weights ", self.torch_network.linear1.bias)
        # print(" Final weights ", self.torch_network.linear1.weight)
        plot_decision_boundary(self.torch_network, self.X, self.y, is_torch=True)


def plot_decision_boundary(network, X, y, is_torch=False):
    x_values = np.linspace(0, 1, 100)
    y_values = np.linspace(0, 1, 100)
    xx, yy = np.meshgrid(x_values, y_values)
    coordinates = np.column_stack((xx.flatten(), yy.flatten()))

    predictions = []
    for coordinate in coordinates:
        if is_torch:
            coordinate = torch.from_numpy(coordinate).float()
            predictions.append(network(coordinate).detach().numpy())
        else:
            predictions.append(network.forward(coordinate))

    predictions = np.array(predictions).reshape(xx.shape)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', label='Training Set')
    # circle = plt.Circle(np.array([0.5, 0.5]), 1 / (np.sqrt(2) * np.pi), color='gray', alpha=0.3)
    # plt.gca().add_patch(circle)

    plt.contourf(xx, yy, predictions, cmap='bwr', alpha=0.8)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Decision Boundary')
    plt.show()


if __name__ == "__main__":
    unittest.main()
