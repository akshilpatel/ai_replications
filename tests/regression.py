from   sklearn.linear_model import LinearRegression as SKLinearRegression
from   replication.regression.linear import LinearRegression
import pandas as pd
import numpy as np

def load_boston():
    # sklearn.dataset.load_boston deprecated
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df   = pd.read_csv(data_url, sep = r"\s+", skiprows = 22, header = None)
    X        = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    y        = raw_df.values[1::2, 2]
    return X, y
    
def test_linear_coef():
    X, y   = load_boston()
    model1 = SKLinearRegression().fit(X, y)
    model2 = LinearRegression().fit(X, y)
    assert np.isclose(model1.coef_, model2.w).all()