import numpy as np

class Logistic_Regression:
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, Y):
        self.m, self.n = X.shape # m is my number of entries and n is number of X features
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y


        #To implement gradient descent
        for i in range(self.iterations):
            self.update_weights()

   
    def update_weights(self):
        Y_pred = self.predict(self.X)
        #calculating gradient descent
        dw = ((self.X.T).dot(Y_pred - self.Y))/self.m
        db = np.sum(Y_pred- self.Y)/self.m

        #updating weights
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def predict(self, X):
        Z = X.dot(self.w) + self.b
        return 1/(1 + np.exp(-Z))

    def predict_classes(self, X, threshold=0.5):
        return (self.predict(X) >= threshold).astype(int)
