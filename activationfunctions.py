import numpy as np
from module import Module

class Activation(Module):
    def __init__(self) -> None:
        self.activation = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError

    def backward(self, previous_gradient):
        raise NotImplementedError



class Sigmoid(Activation):
    def __init__(self) -> None:
        self.activation = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.activation = 1 / (1 + np.exp(-x))
        return self.activation

    def backward(self, previous_gradient):
        return previous_gradient * self.activation * (1 - self.activation)


class ReLU(Activation):
    def __init__(self) -> None:
        self.activation = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.activation = np.maximum(0, x)
        return self.activation

    def backward(self, previous_gradient):
        return previous_gradient * np.where(self.activation > 0, 1, 0)
