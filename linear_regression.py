import numpy as np
from matplotlib import pyplot as plt
from optimizers import Optimizer
from error_checking import ErrorChecker

import time

from types import FunctionType


class LinearRegression:
    '''
    Класс реализующий линейную регрессию
    '''

    dots: np.ndarray

    optimizer : FunctionType
    
    weight : float
    bias   : float

    learning_rate : float

    def __init__(self, dots: np.ndarray, optimizer : str, max_iterations : int, learning_rate : float) -> None:
        self.dots = dots

        self.weight = np.random.normal(loc=0, scale=0.01)
        self.bias   = np.random.normal(loc=0, scale=0.01)

        self.learning_rate = learning_rate

        self.optimizer = LinearRegression.define_optimizer(optimizer)

        self.max_iterations = max_iterations

    
    def run(self):
        x = self.dots[:, 0]
        y = self.dots[:, 1]
        
        mse = ErrorChecker.MSE

        plt.ion()

        losses = np.array([])
        for iter in range(self.max_iterations):
            prediction = self.prediction(x)
            self.weight, self.bias = self.optimizer(dots = self.dots, weight = self.weight, bias = self.bias, batch_size = int(len(self.dots) / 2),  learning_rate = self.learning_rate)
            loss = mse(dots = self.dots, weight = self.weight, bias = self.bias)
            losses = np.append(losses, loss)
            print(f"loss (iteration #{iter}): {loss}")
            self.visualize(x = x, y = y, prediction = prediction)

        plt.ioff()
        plt.show()

        print(f"Коеффицент детерминации: {self.determination_score(prediction = prediction, y = y)}")


    def prediction(self, x):
        return np.dot(x, self.weight) + self.bias
    

    def determination_score(self, prediction, y : np.ndarray):
        return 1 - np.mean(np.power((y - prediction), 2)) / np.mean(np.power((y - np.mean(y)), 2))


    def visualize(self, x, y, prediction):
        plt.clf()
        plt.scatter(x, y, marker='o', alpha=0.6)
        plt.plot(x, prediction, 'r')
        plt.title(label = "Y = {weigth}, X + {bias}".format(weigth=self.weight, bias=self.bias), fontsize = 10, color = "0.8")
        plt.draw()
        plt.gcf().canvas.flush_events()
        time.sleep(0.01)

    
    @staticmethod
    def define_optimizer(optimizer : str) -> FunctionType:
        if optimizer == "SGD":
            return Optimizer.SGD


X = np.array([1, 1.2, 1.6, 1.78, 2, 2.3, 2.4, 3, 3.3, 4, 4.1, 4.12, 4.34, 5, 5.3, 5.6, 6])
Y = np.array([0.8, 1, 0.9, 1.0, 1.2, 1.1, 1.6, 1.7, 2.0, 2.1, 2.15, 2.22, 2.45, 2.6, 2.12, 2.45, 2.3])

dots_sample = np.column_stack((X, Y))

Lg = LinearRegression(dots=dots_sample, optimizer="SGD", max_iterations=150, learning_rate=0.01)
Lg.run()
