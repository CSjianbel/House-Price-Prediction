import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.intercept = None
        self.coefficients = None

    def fit(self, X: np.array, y: np.array):
        mean_x = np.mean(X)
        mean_y = np.mean(y)

        numerator = 0
        denominator = 0
        n = len(X)

        for i in range(n):
            numerator += (X[i] - mean_x) * (y[i] - mean_y)
            denominator += (X[i] - mean_x) ** 2

        self.coefficients = numerator / denominator
        self.intercept = mean_y - (self.coefficients * mean_x)

    def predict(self, X: np.array) -> float:
        return np.dot(X, self.coefficients) + self.intercept

    def visualize(self, X: np.array, y: np.array, x_label: str, y_label: str) -> None:
        plt.scatter(X, y, color='blue')

        if self.coefficients is not None and self.intercept is not None:
            x_values = np.linspace(np.min(X), np.max(X), 100)
            y_values = self.coefficients * x_values + self.intercept
            plt.plot(x_values, y_values, color='red')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title('House Price Prediction')
        plt.grid(True)
        plt.show()
