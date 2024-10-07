import numpy as np

class LinearRegression:
    def __init__(self, learning_rate = 0.0001, epochs = 1000):
        """Initialize the model parameters and hyperparameters
        Parameters are learned coefficients of the model that are optimized during training, 
        while **hyperparameters** are external settings that govern the 
        training process and are set before training begins."""

        self.learning_rate = learning_rate # Hyperparameter
        self.epochs = epochs # Hyperparameter
        self.weights = 0 # Parameter
        self.bias = 0 # Paraeter
    def predict(self, X):
        """Make predictions using the linear model y = mX + b."""
        return (np.dot(X, self.weights) + self.bias)

    def compute_loss(self, X, y):
        """loss function measures the error for a single data point, while a cost
        function measures the error for the entire training set"""
        preds = self.predict(X)
        mse = np.mean((y - preds) ** 2)
        return mse
    def fit(self, X, y):
        """Train the model using Gradient Descent"""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            preds = self.predict(X)
            errors = preds - y

            # Calculate gradients for weights(slope) and bias(intercept)
            weight_gradient = (2/n_samples) * np.dot(X.T, errors)
            bias_gradient = (2/n_samples) * np.sum(errors)

            # Update Parameters 
            self.weights -= self.learning_rate * weight_gradient
            self.bias -= self.learning_rate * bias_gradient

            if epoch % 100 == 0:
                loss = self.compute_loss(X, y)
                print(f"Epoch {epoch}: Loss = {loss}")

    def evaluate(self, X_test, y_test):
        """Evaluate the model performance on new test data."""
        predictions = self.predict(X_test)
        mse = self.compute_loss(X_test, y_test)
        print(f"Evaluation MSE: {mse}")
        return mse        