"""
Author: Matthias Willer 2017
"""

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def hist_plot(x):
    num_bins = 50

    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(x, num_bins, normed=1)

    # add a 'best fit' line
    #y = mlab.normpdf(bins, 4, 2)
    #ax.plot(bins, y, '--')

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
