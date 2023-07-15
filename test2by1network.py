import unittest

import numpy as np

from linearlayer import LinearLayer
from lossfunctions import MSE
from sequential import Sequential


class Test2x1Network(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.network = Sequential([LinearLayer(2, 1)])
        self.network.get_layers()[0].set_parameters(
            np.array([[1, 2]]), np.array([[1]]))

    def test_forward_pass_succeeds(self):
        """
        We are performing one forward step and basically doing the calculation:
        y = x * w.T + b
        y = [0,1] * [1,2].T + [1]
        y = [3]
        """
        data = np.array([[0, 1]])
        result = self.network.forward(data)
        self.assertTrue(np.array_equal(result, np.array([[3]])))

    def test_setting_parameters_of_wrong_shape_fails(self):
        with self.assertRaises(AssertionError):
            self.network.get_layers()[0].set_parameters(
                np.array([[1]]), np.array([[1, 2]]))

    def backward_pass_calculates_correct_gradients(self):
        input = np.array([[1, 2]])
        target = np.array([[1]])
        output = self.network.forward(input)
        loss = self.loss_function(output, target)
        mse_derivative = self.loss_function.backward(output, target)
        linear_gradient = self.network.backward(mse_derivative)
        self.assertTrue(np.array_equal(linear_gradient, np.array([[3, 3]])))


if __name__ == "__main__":
    unittest.main()
