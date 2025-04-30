import warnings

import numpy as np


def sigmoid(x, x0, k, b):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = (1 / (1 + np.exp(-k * (x - x0)))) + b
    return out


def fit_sigmoid(xdata, ydata, p0=None):
    from scipy.optimize import curve_fit

    popt, pcov = curve_fit(sigmoid, xdata, ydata, p0)
    return popt
