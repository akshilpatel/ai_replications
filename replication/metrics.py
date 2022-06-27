from   replication.abstract import Generic
import numpy as np

class RegressionMetrics(Generic):
    """
    Regression Metrics Class
    
    Computes desired metrics given ground truth and predictions.

    Parameters
    ----------
        y : numpy.ndarray
            Array with shape (N,) containing target values.

        y_hat : numpy.ndarray
            Array with shape (...,N) containing predicted values.

        *metrics : str
            String of metrics to use. If None, all metrics are computed.
    """
    def __init__(self, y, y_hat, *metrics):
        super().__init__(locals())
        y, y_hat = map(np.array, [y, y_hat])
        if len(metrics) == 0:
            self._metric = {metric : func(y, y_hat) for metric, func in _metrics.items()}
        else:
            metrics      = set(metrics) | {'count'}
            self._metric = {metric : _metrics[metric](y, y_hat) for metric in metrics}


    def __call__(self, *keys, func = None):
        ret  = None
        if keys:
            if func:
                ret = {key : func(np.array(list(self._metric[key].values()))) for key in keys}
            else:
                ret = {key : self._metric[key] for key in keys}
        elif func:
            ret = {key : func(np.array(list(value.values()))) for key, value in self._metric.items()}
        else:
            ret = self._metric

        if ret:
            if len(ret) == 1:
                return ret[list(ret)[0]]

        return ret

    def __repr__(self):
        inner = ', '.join(f'{key} : {value:.3f}' for key, value in self._metric.items())
        return f'RegressionMetrics({inner})'

    def __getitem__(self, name):
        return self._metric[name]

def mean_squared_error(y, y_hat):
    return np.mean(np.square(y - y_hat))

def mean_absolute_error(y, y_hat):
    return np.mean(np.fabs(y - y_hat))

def root_mean_squared_error(y, y_hat):
    return np.sqrt(mean_squared_error(y, y_hat))

def huber_loss(y, y_hat, delta = 1):
    d    = y - y_hat
    a    = np.fabs(d)
    mask = a < delta
    loss = np.square(d[np.where(mask)]).sum() / 2
    loss += delta * (a[np.where(~mask)] - delta / 2).sum()
    return loss / len(y)

_metrics = {'mean_squared_error'      : mean_squared_error,
            'mean_absolute_error'     : mean_absolute_error,
            'root_mean_squared_error' : root_mean_squared_error,
            'huber_loss'              : huber_loss}
