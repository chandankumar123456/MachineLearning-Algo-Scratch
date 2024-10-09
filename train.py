import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X = X.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

fig = plt.figure(figsize=(8, 6))
plt.scatter(X, y, color="b", marker="o", s=30)
plt.title("Generated Data")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

reg = LinearRegression(learning_rate=0.01)
# reg.fit(X_train, y_train)
reg.fit1(X_train, y_train)
preds = reg.predict(X_test)

reg.evaluate(X_test, y_test)

fig = plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, color='blue', s=10, label='Training data')
plt.scatter(X_test, y_test, color='orange', s=10, label='Test data')
plt.plot(X_test, preds, color='black', linewidth=2, label='Prediction')
plt.title("Linear Regression Fit")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()