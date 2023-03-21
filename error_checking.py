import numpy as np

class ErrorChecker:

    @staticmethod
    def MSE(dots : np.ndarray, weight : float, bias : float) -> float:
        
        x = dots[:, 0]
        y = dots[:, 1]

        return np.mean(np.power(y - np.dot(x, weight) - bias, 2))