from module import Module


class Sequential(Module):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def get_parameters(self):
        parameters = []
        for layer in self.layers:
            parameters.append(layer.get_parameters())
        return parameters

    def get_layers(self):
        return self.layers
