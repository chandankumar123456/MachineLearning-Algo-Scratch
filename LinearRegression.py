import numpy as np

class LinearRegressioN:
    def __init__(self, learning_rate = 0.0001, epochs = 1000):
        """Initialize the model parameters and hyperparameters
        Parameters are learned coefficients of the model that are optimized during 
        training, while **hyperparameters** are external settings that govern the 
        training process and are set before training begins."""

        self.learning_rate = learning_rate # Hyperparameter
        self.epochs = epochs # Hyperparameter
        self.weights = 0 # Parameter
        self.bias = 0 # Paraeter
    def predict(self, X):
        """Make predictions using the linear model y = mX + b."""
        return ((self.weights * X) + (self.bias))
    def compute_loss(self, X, y):
        """loss function measures the error for a single data point, while a cost
        function measures the error for the entire training set"""
        preds = self.predict(X)
        mse = np.mean((y - preds) ** 2)
        return mse
    def fit(self, X, y):
        """Train the model using Gradient Descent"""