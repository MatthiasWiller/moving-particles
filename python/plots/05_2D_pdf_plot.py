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
# Version 2017-10
# ---------------------------------------------------------------------------
# References:
# 1."Simulation and the Monte Carlo method"
#    Rubinstein and Kroese (2017)
# ---------------------------------------------------------------------------
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import utilities.plots as uplt
import utilities.stats as ustat

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = 22
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

# load data
proposal = 'cs'
# proposal = 'mh'
theta = np.load('python/data/samples_2D_pdf_N1e5_'+ proposal +'.npy')


# define target pdf

# target pdf {f(x,y) = 1/20216.335877 * exp(-(x*x*y*y + x*x + y*y - 8*x - 8*y)/2)}
c = 1/20216.335877
target_PDF = lambda x: c * np.exp(-(x[0]*x[0]*x[1]*x[1] + x[0]*x[0] + x[1]*x[1] - 8*x[0] - 8*x[1])/2)




# OUTPUT
print('E[X1] =', round(theta[0,:].mean(), 5))
print('E[X2] =', round(theta[1,:].mean(), 5))

ustat.get_acceptance_rate(theta)

# plot samples
#uplt.plot_hist(theta[0,:], target_PDF, 2)
#uplt.plot_scatter_with_contour(theta, target_PDF)
# uplt.plot_mixing(theta[0, :1000])
uplt.plot_scatter_with_hist(theta[:, :5000], target_PDF)
# uplt.plot_autocorr(theta[0,:], 50, 1)
# uplt.plot_autocorr(theta[1,:], 50, 2)

# PLOTS

# Mixing
matplotlib.rcParams['font.size'] = 26
n_samples = len(theta)

plt.figure()
plt.plot(theta[0,:1000], color='navy')

# set labels
plt.xlabel(r'Number of samples, $n$')
plt.ylabel(r'$x_1$')
plt.xlim(0, n_samples)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

plt.tight_layout()
plt.savefig('plot_mixing_' + proposal + '.pdf', format='pdf', dpi=50, bbox_inches='tight')

# Autocorrelation 1
matplotlib.rcParams['font.size'] = 26

# compute sample autocorrelation
lag = 50
x = theta[0,:]
n_samples   = len(x)
rho         = np.zeros(lag, float)
sigma2      = x.var()
x           = x - x.mean()

for k in range(0, lag):
    temp = 0
    for t in range(0, n_samples - k):
        temp += x[t] * x[t+k]

    rho[k] = (1/sigma2) * (1/(n_samples - k)) * temp

# plot results
plt.figure()
plt.plot(rho[:22], '.')

# set labels
plt.xlabel(r'Lag, $k$')
plt.ylabel(r'$\hat{R}(k)')
plt.xlim(0, lag)
plt.ylim(0, 1)
plt.yticks([0, 0.5, 1])
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.tight_layout()
plt.savefig('plot_autocorr_1_'+proposal+'.pdf', format='pdf', dpi=50, bbox_inches='tight')


# Autocorrelation 2
matplotlib.rcParams['font.size'] = 26

# compute sample autocorrelation
lag = 50
x = theta[1,:]
n_samples   = len(x)
rho         = np.zeros(lag, float)
sigma2      = x.var()
x           = x - x.mean()

for k in range(0, lag):
    temp = 0
    for t in range(0, n_samples - k):
        temp += x[t] * x[t+k]

    rho[k] = (1/sigma2) * (1/(n_samples - k)) * temp

# plot results
plt.figure()
plt.plot(rho[:22], '.')

# set labels
plt.xlabel(r'Lag, $k$')
plt.ylabel(r'$\hat{R}(k)')
plt.xlim(0, lag)
plt.ylim(0, 1)
plt.yticks([0, 0.5, 1])
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.tight_layout()
plt.savefig('plot_autocorr_2_'+proposal+'.pdf', format='pdf', dpi=50, bbox_inches='tight')

plt.show()
