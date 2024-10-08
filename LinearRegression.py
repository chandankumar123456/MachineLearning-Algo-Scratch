import numpy as np
import matplotlib.pyplot as plt
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

        plt.figure(figsize=(10, 6))  # Create a new figure for visualization

        for epoch in range(self.epochs):
            preds = self.predict(X)
            errors = preds - y

            # Calculate gradients for weights(slope) and bias(intercept)
            weight_gradient = (2/n_samples) * np.dot(X.T, errors)
            bias_gradient = (2/n_samples) * np.sum(errors)

            # Update Parameters 
            self.weights -= self.learning_rate * weight_gradient
            self.bias -= self.learning_rate * bias_gradient

            # Plotting
            if epoch % 100 == 0:  # Plot every 100 epochs
                plt.scatter(X, y, color='orange', s=10, label='Data')
                plt.plot(X, preds, color='black', linewidth=2, label='Prediction')
                plt.title(f"Epoch {epoch}: Linear Regression Fit")
                plt.xlabel("X")
                plt.ylabel("y")
                plt.legend()
                plt.show()  # Show plot after each 100 epochs

                loss = self.compute_loss(X, y)
                print(f"Epoch {epoch}: Loss = {loss}")

        # Final plot after training
        plt.scatter(X, y, color='orange', s=10, label='Data')
        plt.plot(X, preds, color='black', linewidth=2, label='Final Prediction')
        plt.title("Final Linear Regression Fit")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        plt.show()

    def fit1(self, X, y):
        """Train the model using Gradient Descent"""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Create a grid of subplots, e.g., 10x10 for 100 subplots (for every 10th epoch)
        fig, axs = plt.subplots(10, 10, figsize=(20, 20))  # 10 rows and 10 columns
        fig.suptitle("Linear Regression Fit over Epochs", fontsize=16)

        plot_counter = 0  # To track which subplot to use

        for epoch in range(self.epochs):
            preds = self.predict(X)
            errors = preds - y

            # Calculate gradients for weights(slope) and bias(intercept)
            weight_gradient = (2/n_samples) * np.dot(X.T, errors)
            bias_gradient = (2/n_samples) * np.sum(errors)

            # Update Parameters 
            self.weights -= self.learning_rate * weight_gradient
            self.bias -= self.learning_rate * bias_gradient

            # Plot every 10th epoch
            if epoch % 10 == 0:
                row = plot_counter // 10  # Row index for subplot
                col = plot_counter % 10   # Column index for subplot
                axs[row, col].scatter(X, y, color='orange', s=10)  # Data points
                axs[row, col].plot(X, preds, color='black', linewidth=1)  # Prediction line
                axs[row, col].set_title(f"Epoch {epoch}")
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])

                plot_counter += 1  # Increment to next subplot

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                loss = self.compute_loss(X, y)
                print(f"Epoch {epoch}: Loss = {loss}")

        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout so the title is clear
        plt.show()

    def evaluate(self, X_test, y_test):
        """Evaluate the model performance on new test data."""
        predictions = self.predict(X_test)
        mse = self.compute_loss(X_test, y_test)
        print(f"Evaluation MSE: {mse}")
        return mse        