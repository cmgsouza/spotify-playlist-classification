import numpy as np 

class LogisticRegression:
    def __init__(self, iterations=15000, learning_rate=0.10):
        self.iterations = iterations
        self.learning_rate = learning_rate

    def logistic_function(self, x):
        return 1 / (1 + np.exp(-x))
    
    def fit(self, xtr, ttr):
        xtr = np.concatenate((np.ones((xtr.shape[0], 1)), xtr), axis=1) ##add intercept
        self.weights = np.zeros(xtr.shape[1]) ##initialize weights        
        func = self.logistic_function(np.dot(xtr, self.weights))
        for i in range(self.iterations):
            self.weights -= (np.dot(xtr.T, (func - ttr)) / ttr.size) * self.learning_rate
            func = self.logistic_function(np.dot(xtr, self.weights))
        return self
    
    def predict(self, xte):
        xte = np.concatenate((np.ones((xte.shape[0], 1)), xte), axis=1)
        return self.logistic_function(np.dot(xte, self.weights)).round()