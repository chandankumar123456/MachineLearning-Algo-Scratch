import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import load_breast_cancer
from LogisticRegression import LogisticRegression

bc = load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(lr=0.001)
clf.fit(X_train, y_train)
y_preds = clf.predict(X_test)

print(classification_report(y_test, y_preds))
print("Accuracy:", accuracy_score(y_test, y_preds))

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

x_values = np.linspace(-10, 10, 100)
sigmoid_values = sigmoid(x_values)

plt.figure(figsize=(10, 6))
plt.plot(x_values, sigmoid_values, label='Sigmoid Function', color='blue')

linear_preds = np.dot(X_test, clf.weight) + clf.bias
predicted_probabilities = sigmoid(linear_preds)

plt.scatter(linear_preds, predicted_probabilities, c=y_test, cmap='bwr', alpha=0.5, edgecolors='k', label='Data Points')
plt.axhline(0.5, color='red', linestyle='--', label='Decision Boundary (0.5)')
plt.xlabel('Linear Predictions')
plt.ylabel('Sigmoid Output')
plt.title('Sigmoid Function with Data Points')
plt.legend()
plt.grid()
plt.show()
