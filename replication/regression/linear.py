from replication.abstract import Model
from replication.transforms import transforms
import numpy as np

class LinearRegression(Model):

    def __init__(self, fit_intercept = True, transform = None):
        super().__init__(locals())

    def fit(self, X, y):
        Xc = X.copy()
        if self.transform:
            self.__transform = transforms[self.transform].fit(X)
            X                = self.__transform.transform(X)
        if self.fit_intercept:
            X                = np.insert(X, 0, 1, 1)
        w = np.linalg.solve(X.T @ X, X.T @ y)
        if self.fit_intercept:
            self.b = w[0]
            self.w = w[1:]
        else:
            self.b = 0
            self.w = w
        return self

    def predict(self, X):
        if self.transform:
            X = self.__transform.transform(X)
        return self.b + X @ self.w

    def predict_proba(self, X):
        raise Exception()

    def predict_log_proba(self, X):
        raise Exception()
