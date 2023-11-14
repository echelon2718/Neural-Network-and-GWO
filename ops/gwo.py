import numpy as np

class GWO:
    def __init__(self, nn, X, y, num_wolves=10, max_iter=1000):
        self.nn = nn
        self.X = X
        self.y = y
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.dim = len(nn.pack_params())
        self.wolves = np.random.randn(num_wolves, self.dim)
        self.alpha = np.zeros(self.dim)
        self.beta = np.zeros(self.dim)
        self.delta = np.zeros(self.dim)
        self.a = 2  # Parameter for GWO, tune this as you like

    def optimize(self):
        for iter in range(self.max_iter):
            # Update the leading wolves (alpha, beta, delta)
            self.update_leading_wolves()

            # Update position of each wolf
            for i in range(self.num_wolves):
                for d in range(self.dim):
                    A1 = 2 * self.a * np.random.rand() - self.a
                    C1 = 2 * np.random.rand()
                    D_alpha = abs(C1 * self.alpha[d] - self.wolves[i, d])
                    X1 = self.alpha[d] - A1 * D_alpha

                    A2 = 2 * self.a * np.random.rand() - self.a
                    C2 = 2 * np.random.rand()
                    D_beta = abs(C2 * self.beta[d] - self.wolves[i, d])
                    X2 = self.beta[d] - A2 * D_beta

                    A3 = 2 * self.a * np.random.rand() - self.a
                    C3 = 2 * np.random.rand()
                    D_delta = abs(C3 * self.delta[d] - self.wolves[i, d])
                    X3 = self.delta[d] - A3 * D_delta

                    self.wolves[i, d] = (X1 + X2 + X3) / 3

            self.a -= 2 / self.max_iter

    def update_leading_wolves(self):
        fitness_values = np.array([self.fitness(self.wolves[i]) for i in range(self.num_wolves)])
        sorted_indices = np.argsort(fitness_values)

        self.alpha = self.wolves[sorted_indices[0]]
        self.beta = self.wolves[sorted_indices[1]]
        self.delta = self.wolves[sorted_indices[2]]

    def fitness(self, wolf):
        self.nn.unpack_params(wolf)
        y_pred = self.nn.forward(self.X)  # Pass the training data
        return self.nn.loss(self.y, y_pred)