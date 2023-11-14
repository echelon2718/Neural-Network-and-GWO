import numpy as np
from sklearn.metrics import accuracy_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0, keepdims=True)

def cross_entropy(y_true, y_pred):
    m = y_true.shape[1]
    cost = -1/m * np.sum(y_true * np.log(y_pred))
    return cost

def predict(nn, X):
    return nn.predict(X)