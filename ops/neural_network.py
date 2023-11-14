import numpy as np
from ops.base_function import sigmoid, softmax, cross_entropy

class NeuralNetwork:
    def __init__(self, x, layers):
        self.x = x
        self.layers = layers
        self.weights = []
        self.biases = []
        self.initialize_weights_and_biases()

    def initialize_weights_and_biases(self):
        for item in range(0, len(self.layers) - 1):
            self.weights.append(np.random.randn(self.layers[item+1], self.layers[item]) * 0.01)
            self.biases.append(np.random.randn(self.layers[item+1], 1) * 0.01)

    def forward(self, X=None):
        if X is None:
            X = self.x

        A = X

        for params in range(len(self.layers) - 2):
            Z = np.matmul(self.weights[params], A) + self.biases[params]
            A = sigmoid(Z)

        Z = np.matmul(self.weights[-1], A) + self.biases[-1]
        A = softmax(Z)

        return A

    def loss(self, y_true, y_pred): # Vanilla cross entropy loss
        return cross_entropy(y_true, y_pred)

    def pack_params(self): # Utility for packing weight parameters
        return np.concatenate([w.ravel() for w in self.weights] + [b.ravel() for b in self.biases])

    def unpack_params(self, params): # Utility for unpacking weight parameters
        id = 0
        for i in range(len(self.layers) - 1):
            weight_shape = (self.layers[i + 1], self.layers[i])
            bias_shape = (self.layers[i + 1], 1)
            self.weights[i] = params[id:id + np.prod(weight_shape)].reshape(weight_shape)
            id += np.prod(weight_shape)
            self.biases[i] = params[id:id + np.prod(bias_shape)].reshape(bias_shape)
            id += np.prod(bias_shape)
    
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis = 0)
    