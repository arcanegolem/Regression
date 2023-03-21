import numpy as np
from error_checking import ErrorChecker


class Optimizer:

    @staticmethod
    def SGD(dots : np.ndarray, weight : float, bias : float, batch_size : int, learning_rate : float):
        weight_derivative = 0
        bias_derivative   = 0

        shuffled_dots = dots.copy()
        np.random.shuffle(shuffled_dots)

        batch = shuffled_dots[ : batch_size]

        x = batch[:, 0]
        y = batch[:, 1]
        
        for i in range(len(batch)):
            weight_derivative += -2 * x[i] * (y[i] - (weight * x[i] + bias)) / len(batch)
            bias_derivative += -2 * (y[i] - (weight * x[i] + bias)) / len(batch)
        
        new_weigth = weight - learning_rate * weight_derivative
        new_bias   = bias   - learning_rate * bias_derivative

        return (new_weigth, new_bias)