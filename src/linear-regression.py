import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, method='gradient_descent'):
        self.method = method
        self.intercept = None
        self.coefficients = None

    def fit(self, X: np.array, y: np.array):
        if self.method == 'simple':
            self._fit_simple_linear_regression(X, y)
        elif self.method == 'gradient_descent':
            self._fit_gradient_descent(X, y)
        else:
            raise ValueError("Invalid method. Choose 'simple' or 'gradient_descent'.")

    def _fit_simple_linear_regression(self, X: np.array, y: np.array):
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

    def _fit_gradient_descent(self, X: np.array, y: np.array, learning_rate: float=0.01, num_iterations: int=1000):
        num_samples = len(X)
        self.coefficients = np.zeros(X.shape[1])
        self.intercept = 0

        for _ in range(num_iterations):
            y_predicted = np.dot(X, self.coefficients) + self.intercept

            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            self.coefficients -= learning_rate * dw
            self.intercept -= learning_rate * db

    def predict(self, X: np.array) -> float:
        return np.dot(X, self.coefficients) + self.intercept

    def visualize_linear_regression(self, X: np.array, y: np.array, x_label: str, y_label: str) -> None:
        plt.scatter(X, y, color='blue', label='Data points')

        if self.coefficients is not None and self.intercept is not None:
            x_values = np.linspace(np.min(X), np.max(X), 100)
            y_values = self.coefficients * x_values + self.intercept
            plt.plot(x_values, y_values, color='red', label='Best-fit line')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title('House Price Prediction')
        plt.legend()
        plt.grid(True)
        plt.show()
