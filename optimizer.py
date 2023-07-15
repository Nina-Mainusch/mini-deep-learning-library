import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, network):
        layers = network.get_layers()
        for layer in layers:
            if layer.__class__.__base__.__name__ != "Activation":
                weights, bias = layer.get_parameters()
                gradient_weights, gradient_bias = layer.get_gradients()

                # print(gradient_weights, weights)
                # print(gradient_bias, bias)

                weights -= self.lr * gradient_weights
                bias -= self.lr * gradient_bias

                layer.set_parameters(weights, bias)

    def zero_grad(self, network):
        layers = network.get_layers()
        for layer in layers:
            if layer.__class__.__base__.__name__ != "Activation":
                layer.gradient_weights = np.zeros_like(layer.gradient_weights)
                layer.gradient_bias = np.zeros_like(layer.gradient_bias)
