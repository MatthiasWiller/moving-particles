"""
Author: Matthias Willer 2017
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps

import plots.user_plot as uplt
import algorithms.metropolis_hastings as mh

# INPUT 

# target pdf (two weighted gaussian)
def target_PDF(x):
    c = 1/20216.335877
    # f(x,y) = 1/20216.335877 * exp(-(x*x*y*y + x*x + y*y - 8*x - 8*y)/2)
    return c * np.exp(-(x[0]*x[0]*x[1]*x[1] + x[0]*x[0] + x[1]*x[1] - 8*x[0] - 8*x[1])/2)

# proposal pdf 
def sample_prop_PDF(mu):
    mu = [0, 0]
    cov = [[4, 0],[0, 4]]
    #return sps.multivariate_normal(mu, cov, 1)
    return sps.multivariate_normal.rvs(mu, cov, 1)

# proposal pdf
def f_prop_PDF(x, param):
    mu = [0, 0]
    cov = [[4, 0],[0, 4]]
    return sps.multivariate_normal.pdf(x, mu, cov)

np.random.seed(1)
initial_theta = [1.5, 1.5]         # initial theta
n_samples = 5000         # number of samples
burnInFraction = 0.1     # defines burn-in-period of samples
lagPeriod = 1              # only log every n-th value


# apply mcmc-method
theta = mh.metropolis_hastings(initial_theta, n_samples, target_PDF, sample_prop_PDF, f_prop_PDF, burnInFraction, lagPeriod)

# OUTPUT

print('E[X1] =', round(theta[0,:].mean(), 5))
print('E[X2] =', round(theta[1,:].mean(), 5))

# plot samples
#uplt.plot_hist(theta[0,:], target_PDF, 2)
uplt.plot_scatter_with_contour(theta, target_PDF)
uplt.plot_mixing(theta[0,:])
#uplt.plot_surface_custom(target_PDF)
uplt.plot_autocorr(theta[0,:], 400)
uplt.plot_scatter_with_hist(theta, target_PDF)
plt.show()
