"""
# ---------------------------------------------------------------------------
# Metropolis-Hastings algorithm 2D example: Example 6.4 Ref. [1]
# ---------------------------------------------------------------------------
# Created by:
# Matthias Willer (matthias.willer@tum.de)
# Engineering Risk Analysis Group
# Technische Universitat Munchen
# www.era.bgu.tum.de
# ---------------------------------------------------------------------------
# Version 2017-05
# ---------------------------------------------------------------------------
# References:
# 1."Simulation and the Monte Carlo method"
#    Rubinstein and Kroese (2017)
# ---------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt

import utilities.plots as uplt
import utilities.stats as ustat
import algorithms.metropolis_hastings as mh

# define target pdf

# target pdf {f(x,y) = 1/20216.335877 * exp(-(x*x*y*y + x*x + y*y - 8*x - 8*y)/2)}
c = 1/20216.335877
target_PDF = lambda x: c * np.exp(-(x[0]*x[0]*x[1]*x[1] + x[0]*x[0] + x[1]*x[1] - 8*x[0] - 8*x[1])/2)


theta = np.load('python/data/samples_2D_pdf_N1e5.npy')

# OUTPUT
print('E[X1] =', round(theta[0,:].mean(), 5))
print('E[X2] =', round(theta[1,:].mean(), 5))

#ustat.get_acceptance_rate(theta)

# plot samples
#uplt.plot_hist(theta[0,:], target_PDF, 2)
#uplt.plot_scatter_with_contour(theta, target_PDF)
uplt.plot_mixing(theta[0, :1000])
# uplt.plot_scatter_with_hist(theta[:, :5000], target_PDF)
uplt.plot_autocorr(theta[0,:], 50, 1)
# uplt.plot_autocorr(theta[1,:], 50, 2)
plt.show()
