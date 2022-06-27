from   replication.abstract import Transform
import numpy as np

class StandardScalar(Transform):
    """
    Transforms data such that each feature has mean 0 and standard deviation of 1.
    """
    def __init__(self):
        super().__init__(locals())

    def fit(self, X, y = None):
        self.mu = X.mean(axis = -1, keepdims = True)
        self.sd = X.std (axis = -1, keepdims = True)
        return self

    def fit_transform(self, X, y = None):
        self.fit(X)
        return self.transform(X)
    
    def transform(self, X):
        return (X - self.mu) / self.sd

    def inverse_transform(self, Z):
        return Z * self.sd + self.mu

class Normalize(Transform):
    """
    Transforms data such that each feature has mean 0 and a 2-norm of 1.
    """
    def __init__(self):
        super().__init__(locals())

    def fit(self, X, y = None):
        self.mu = X.mean(axis = -1, keepdims = True)
        self.l2 = np.sqrt(np.square(X - self.mu).sum(axis = -1, keepdims = True))
        return self

    def fit_transform(self, X, y = None):
        self.fit(X)
        return self.transform(X)
    
    def transform(self, X):
        return (X - self.mu) / self.l2

    def inverse_transform(self, Z):
        return Z * self.l2 + self.mu

class MinMaxScalar(Transform):
    """
    Transforms data such that each feature is between 0 and 1.
    """
    def __init__(self):
        super().__init__(locals())

    def fit(self, X, y = None):
        self.lo  = X.min(axis = -1, keepdims = True)
        self.hi  = X.max(axis = -1, keepdims = True)
        self.ptp = self.hi - self.lo
        return self

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)
    
    def transform(self, X):
        return (X - self.lo) / self.ptp

    def inverse_transform(self, Z):
        return Z * self.ptp + self.lo


transforms = {'StandardScalar' : StandardScalar,
              'Normalize'      : Normalize,
              'MinMaxScalar'   : MinMaxScalar}