import numpy as np
from matplotlib import pyplot as plt

class LinearRegression:
    '''
    Класс реализующий линейную регрессию
    '''

    dots: np.ndarray

    def __init__(self, dots: np.ndarray) -> None:
        self.dots = dots

    
    def run(self):
        pass

    
    def visualize(self):
        fig = plt.figure(figsize=(10, 5))

        axes_orig = fig.add_subplot(1, 2, 1)
        axes_orig.scatter(self.dots[:, 0], self.dots[:, 1], marker="o")
        axes_orig.set_title("Dots placed")

        # TODO: Add visualization for end result

        plt.show()


Lg = LinearRegression(dots=np.array([[1, 1000], [2, 1500], [3, 2000], [4, 2500]]))
Lg.visualize()

