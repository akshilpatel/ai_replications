from abc import ABC, abstractclassmethod

base_exceptions = ['self', '__class__']

class Generic():
    """
    Generic Base Class.
    """
    def __init__(self, params, exceptions = []):
        self.__store__(params, exceptions)

    def __store__(self, params, exceptions = []):
        exceptions   += base_exceptions
        self.__params = []
        for key, value in params.items():
            if key not in exceptions and key:
                setattr(self, key, value)
                self.__params.append(key)

        self.__name__ = self.__class__.__name__

    def __repr__(self):
        inner = ', '.join([f'{key} = {getattr(self, key)}' for key in self.__params])
        return f'{self.__name__}({inner})'

class Model(Generic, ABC):
    """
    Generic Model Class with abstract class methods.

    To be used for main machine learning models.
    """
    def __init__(self, params):
        Generic.__init__(self, params)
        ABC.__init__(self)

    @abstractclassmethod
    def fit(self, X, y):
        pass

    @abstractclassmethod
    def predict(self, X):
        pass

    @abstractclassmethod
    def predict_proba(self, X):
        pass

    @abstractclassmethod
    def predict_log_proba(self, X):
        pass


class Transform(Generic, ABC):
    """
    Generic Transform Class with abstract class methods.

    To be used for Transformation Class Objects such as StandardScalar.
    """
    def __init__(self, params):
        Generic.__init__(self, params)
        ABC.__init__(self)

    @abstractclassmethod
    def fit(self, X, y = None):
        pass

    @abstractclassmethod
    def fit_transform(self, X, y = None):
        pass

    @abstractclassmethod
    def transform(self, X):
        pass

    @abstractclassmethod
    def inverse_transform(self, Z):
        pass