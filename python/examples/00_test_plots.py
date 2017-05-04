"""
Author: Matthias Willer 2017
"""
import time as timer

print("RUN 00_test_plots.py")
startTime = timer.time()
print('\n\n> START Importing')

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as npdf

import plots.user_plot as uplt

# INPUT 
print("> Time needed for Importing =", round(timer.time() - startTime, 2), "s")
startTime = timer.time()
print('\n\n> START Sampling')

n_samples = 500
mu = 3
sigma = 3

# target pdf 2D (Example from 05_2D_pdf.py)
def target_PDF_2D(x):
    c = 1/20216.335877
    # f(x,y) = 1/20216.335877 * exp(-(x*x*y*y + x*x + y*y - 8*x - 8*y)/2)
    return c * np.exp(-(x[0]*x[0]*x[1]*x[1] + x[0]*x[0] + x[1]*x[1] - 8*x[0] - 8*x[1])/2)

# target pdf 1D
def target_PDF_1D(x):
    return npdf.norm.pdf(x, mu, sigma)

# theta 2D
theta = np.zeros((2, n_samples), float)
theta[0,:] = npdf.norm.rvs(mu, sigma, n_samples)
theta[1,:] = npdf.norm.rvs(mu, sigma, n_samples)

print("> Time needed for Sampling =", round(timer.time() - startTime, 2), "s")
startTime = timer.time()
print('\n\n> START Plotting')

# OUTPUT

# plot samples
uplt.plot_hist(theta[0,:], target_PDF_1D)
uplt.plot_mixing(theta[0,:])
uplt.plot_autocorr(theta[0,:], n_samples)
uplt.plot_scatter_with_contour(theta, target_PDF_2D)

print("> Time needed for Plotting =", round(timer.time() - startTime, 2), "s")
plt.show()
