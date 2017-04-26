"""
Author: Matthias Willer 2017
"""

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def hist_plot(x, target_PDF=0):
    num_bins = 50

    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(x, num_bins, normed=1)

    # add a 'best fit' line
    if(target_PDF != 0):
        y = target_PDF(bins)
        ax.plot(bins, y, '--')



    ax.set_xlabel('theta')
    ax.set_ylabel('p')
    ax.set_title(r'Histogram of theta')

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()


def n_plot(x):
    fig, ax = plt.subplots()

    ax.plot(x)

    ax.set_xlabel('n')
    ax.set_ylabel('theta')
    ax.set_title(r'Plot of theta')

    fig.tight_layout()

def estimated_autocorrelation(x, n_samples):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """

    n = np.minimum(len(x),n_samples)
    x = x[:n]
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    
    
    fig, ax = plt.subplots()
    
    ax.stem(result)
    ax.set_title(r'Autocorrelation')
    ax.set_xlabel('n')
    ax.set_ylabel('autocorrelation')

    fig.tight_layout()

#P.plot(time,estimated_autocorrelation(x))