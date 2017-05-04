"""
Author: Matthias Willer 2017
"""

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def compute_marginal_PDF(target_PDF, bins, dimension):
    x_from = -10
    x_till = 10
    n_steps = 50.0
    dx = (x_till - x_from) / n_steps
    x = np.linspace(x_from, x_till, n_steps)

    len_x = len(x)
    len_bins = len(bins)

    y = np.zeros((len_bins), float)
    # integrate over x1 to obtain marginal of x0
    if dimension == 0:
        for i in range(0, len_bins):
            for j in range(0, len_x):
                temp_x = bins[i]
                temp_y = x[j]
                y[i] += target_PDF([temp_x, temp_y])
            
            y[i] = y[i]*dx

    # integrate over x2 to obtain marginal of x1
    if dimension == 1:
        for i in range(0, len_bins):
            for j in range(0, len_x):
                temp_y = bins[i]
                temp_x = x[j]
                y[i] += target_PDF([temp_x, temp_y])
            
            y[i] = y[i]*dx
    
    # return y
    return y

def estimate_autocorrelation(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    
    # compute variance and mean-value
    #variance = x.var()
    #x = x-x.mean()

    # compute autocorrelation (for explanation see stackoverflow/wikipedia)
    #r = np.correlate(x, x, mode = 'full')[-n:]
    #assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    #result = r/(variance*(np.arange(n, 0, -1)))

    N = len(x)
    p = np.zeros(N, float)

    sigma_square = x.var()
    mu = x.mean()
    #x = x - x.mean()

    for j in range(0, N):
        temp = 0
        for n in range(1, N-j):
            temp = temp + (x[n]-mu)*(x[n+j]-mu)
        p[j] = 1/sigma_square * 1/(N-j) * temp

    return p