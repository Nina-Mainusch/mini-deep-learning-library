import numpy as np
from module import Module


class MSE(Module):
    def __init__(self) -> None:
        self.input_data = None

    def __call__(self, prediction, target):
        return self.forward(prediction, target)

    def forward(self, prediction, target):
        self.input_data = 0.5 * np.sum((prediction - target) ** 2)
        return self.input_data

    def backward(self, prediction, target):
        return prediction - target
