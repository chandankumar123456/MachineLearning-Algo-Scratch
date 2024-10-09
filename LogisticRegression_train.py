import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import load_breast_cancer

from LogisticRegression import LogisticRegression

bc = load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

clf = LogisticRegression(lr = 0.001)
clf.fit(X_train, y_train)
y_preds = clf.predict(X_test)

print(classification_report(y_test, y_preds))
print(accuracy_score(y_test, y_preds))