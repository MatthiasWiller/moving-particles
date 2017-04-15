"""
Author: Matthias Willer 2017
"""

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as npdf


def fcn(x): 
    mu = 4    # mean
    sigma = 2  # standard deviation
    result = npdf.norm.pdf(x, mu, sigma)
    return result

def hist_plot(x):
    num_bins = 50

    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(x, num_bins, normed=1)

    # add a 'best fit' line
    y = mlab.normpdf(bins, 4, 2)
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


# set seed
np.random.seed(0)

# initial value 
initial_theta = 0.0
n_samples = 500
burninperiod = int (n_samples * 0.2)

# set up theta
theta = np.zeros((n_samples), float)
theta[0] = initial_theta


i = 1

# loop
while i < n_samples:
    theta_star = np.random.uniform(-10.0, 18.0, 1)
    alpha = np.minimum( fcn(theta_star) / fcn(theta[i-1]) ,1)
    
    if (np.random.random([1]) < alpha):
        theta[i] = theta_star
        i+=1
        print("accept!\n")
    else:
        print("reject!\n")


theta_new = theta[burninperiod:]

# plot samples
hist_plot(theta_new)
n_plot(theta_new)
plt.show()

