import numpy as np
import matplotlib.pyplot as plt
import time


class Linear(object):
    '''
    Класс метода численной линейной регрессии
    '''
    Y_: np.ndarray
    X_: np.ndarray
    size_: int
    loss_history: np.ndarray
    pred: np.ndarray
    beta1_: float
    beta2_: float
    EMA1_w: float
    EMA1_b: float
    EMA2_w: float
    EMA2_b: float
    t: int


    def __init__(self,
                 max_iter: int = 100,
                 stop_criteria: bool = True,
                 learning_rate: float = 1 * pow(10, -3),
                 optimizer_name: str = "GD",
                 beta1: float = 0.9,
                 beta2: float = 0.999
                 ):
        self.weight_ = np.random.normal(loc=0.0, scale=0.01)
        self.bias_ = np.random.normal(loc=0.0, scale=0.01)
        self.max_iter_ = max_iter
        self.stop_criteria_ = stop_criteria
        self.learning_rate_ = learning_rate
        self.optimizer_name_ = optimizer_name
        self.beta1_ = beta1
        self.beta2_ = beta2


    def fit(self, X, Y):
        self.X_ = X
        self.Y_ = Y
        self.size_ = len(self.X_)

        if self.optimizer_name_ == 'GD':
            optimizer = self.GD
        else:
            optimizer = self.select_optimizer()

        plt.ion()
        self.loss_history = np.array([])
        for i in range(self.max_iter_):
            self.predict(self.X_)
            optimizer()
            loss = self.MSE()
            self.loss_history = np.append(self.loss_history, loss)
            self.plot()

            if i % 10 == 0:
                print("iter: " + str(i) + " loss: " + str(loss))

        plt.ioff()
        plt.show()
        return self


    def MSE(self):
        return np.mean(np.power((self.Y_ - np.dot(self.X_, self.weight_) - self.bias_), 2))


    def GD(self):
        weight_deriv = 0
        bias_deriv = 0

        for i in range(self.size_):
            weight_deriv += -2 * self.X_[i] * (self.Y_[i] - (self.weight_ * self.X_[i] + self.bias_)) / self.size_
            bias_deriv += -2 * (self.Y_[i] - (self.weight_ * self.X_[i] + self.bias_)) / self.size_

        self.weight_ -= self.learning_rate_ * weight_deriv
        self.bias_ -= self.learning_rate_ * bias_deriv


    def SGD(self):
        pass


    def RMSprop(self):
        weight_deriv = 0
        bias_deriv = 0
        epsilon = pow(10, -8)

        for i in range(self.size_):
            weight_deriv += -2 * self.X_[i] * (self.Y_[i] - (self.weight_ * self.X_[i] + self.bias_)) / self.size_
            bias_deriv += -2 * (self.Y_[i] - (self.weight_ * self.X_[i] + self.bias_)) / self.size_

        self.EMA1_w = self.beta1_ * self.EMA1_w + (1 - self.beta1_) * np.power(weight_deriv, 2)
        self.EMA1_b = self.beta1_ * self.EMA1_b + (1 - self.beta1_) * np.power(bias_deriv, 2)

        self.weight_ -= self.learning_rate_ * weight_deriv / np.sqrt(self.EMA1_w + epsilon)
        self.bias_ -= self.learning_rate_ * bias_deriv / np.sqrt(self.EMA1_b + epsilon)


    def Adam(self):
        weight_deriv = 0
        bias_deriv = 0
        epsilon = pow(10, -8)

        for i in range(self.size_):
            weight_deriv += -2 * self.X_[i] * (self.Y_[i] - (self.weight_ * self.X_[i] + self.bias_)) / self.size_
            bias_deriv += -2 * (self.Y_[i] - (self.weight_ * self.X_[i] + self.bias_)) / self.size_

        self.EMA1_w = self.beta1_ * self.EMA1_w + (1 - self.beta1_) * weight_deriv
        self.EMA1_b = self.beta1_ * self.EMA1_b + (1 - self.beta1_) * bias_deriv
        self.EMA2_w = self.beta2_ * self.EMA2_w + (1 - self.beta2_) * np.power(weight_deriv, 2)
        self.EMA2_b = self.beta2_ * self.EMA2_b + (1 - self.beta2_) * np.power(bias_deriv, 2)

        self.EMA1_w = self.EMA1_w / (1 - np.power(self.beta1_, self.t))
        self.EMA1_b = self.EMA1_b / (1 - np.power(self.beta1_, self.t))
        self.EMA2_w = self.EMA2_w / (1 - np.power(self.beta2_, self.t))
        self.EMA2_b = self.EMA2_b / (1 - np.power(self.beta2_, self.t))

        self.weight_ -= self.learning_rate_ * self.EMA1_w / (np.sqrt(self.EMA2_w) + epsilon)
        self.bias_ -= self.learning_rate_ * self.EMA1_b / (np.sqrt(self.EMA2_b) + epsilon)
        self.t += 1


    def plot(self):
        plt.clf()
        plt.scatter(self.X_, self.Y_, marker='o', alpha=0.8)
        plt.plot(self.X_, self.pred, 'r')
        plt.title('y = ' + str(self.weight_) + ' x + ' + str(self.bias_), fontsize=10, color='0.5')
        plt.draw()
        plt.gcf().canvas.flush_events()
        time.sleep(0.01)


    def predict(self, X):
        self.pred = np.dot(X, self.weight_) + self.bias_
        return self.pred
    

    def select_optimizer(self):
        if self.optimizer_name_ == 'SGD':
            return self.SGD
        elif self.optimizer_name_ == 'Adam':
            self.EMA1_w = 0.0
            self.EMA2_w = 0.0
            self.EMA1_b = 0.0
            self.EMA2_b = 0.0
            self.t = 1
            return self.Adam
        elif self.optimizer_name_ == 'RMSprop':
            self.EMA1_w = 0.0
            self.EMA1_b = 0.0
            return self.RMSprop


X = np.array([1, 1.2, 1.6, 1.78, 2, 2.3, 2.4, 3, 3.3, 4, 4.1, 4.12, 4.34, 5, 5.3, 5.6, 6])
Y = np.array([0.8, 1, 0.9, 1.0, 1.2, 1.1, 1.6, 1.7, 2.0, 2.1, 2.15, 2.22, 2.45, 2.6, 2.12, 2.45, 2.3])

regression = Linear(learning_rate=0.01, max_iter=100, optimizer_name='GD') # learning_rate=0.05, max_iter=100, optimizer_name='Adam'
# regression.fit(X=df_train['square'].to_numpy(), Y=df_train['clusters'].to_numpy())

regression.fit(X=X, Y=Y)
x_loss = np.arange(len(regression.loss_history))

fig, axe = plt.subplots(1, 2, figsize=(15, 6))
axe[0].scatter(X, Y, marker='o', alpha=0.8)
axe[0].plot(X, regression.pred, 'r')
axe[1].plot(x_loss, regression.loss_history)
plt.show()