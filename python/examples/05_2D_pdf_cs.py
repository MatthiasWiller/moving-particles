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
import matplotlib.pyplot as plt
import scipy.stats as scps
# from scipy.special import erf
# from scipy.special import erfinv

import scipy.integrate as integrate
import scipy.interpolate as interpolate

import utilities.plots as uplt
import algorithms.mcmc_metropolis_hastings as mh
import algorithms.mcmc_cond_sampling as cs

# INPUT

np.random.seed(1)


# target pdf {f(x,y) = 1/20216.335877 * exp(-(x*x*y*y + x*x + y*y - 8*x - 8*y)/2)}
c = 1/20216.335877
target_PDF = lambda u: c * np.exp(-(u[0]*u[0]*u[1]*u[1] + u[0]*u[0] + u[1]*u[1] - 8*u[0] - 8*u[1])/2)

# parameters for proposal pdf
sig = 2
mu  = [0, 0]
cov = [[sig**2, 0],[0, sig**2]]

# proposal pdf
sample_prop_PDF = lambda param: scps.norm.rvs(param, sig, 2) # draw the components independently
f_prop_PDF      = lambda x, param: scps.multivariate_normal.pdf(x, param, cov) # PDF


initial_theta   = [1.5, 1.5]     # initial theta
n_samples       = int(1e5)       # number of samples
burnInPeriod    = 5000            # defines burn-in-period of samples
lagPeriod       = 5              # only log every n-th value
rho_k           = 0.8

#test configuration
n_samples    = 1000
burnInPeriod = 100
lagPeriod    = 2


# transformation to/from U-space
phi     = lambda x: scps.norm.cdf(x)
phi_inv = lambda x: scps.norm.ppf(x)


# !! -- define target CDF and inverse of target CDF here -- !!
fxy = lambda x, y: c * np.exp(-(x*x*y*y + x*x + y*y - 8*x - 8*y)/2)

x_line = np.linspace(-5, 10, 100)
y_line = np.linspace(-5, 10, 100)
fx = np.zeros(len(x_line))
fy_giv_x = np.zeros((len(x_line), len(y_line)))

for i in range(0, len(x_line)):
    print('integrating pdf:',i/len(x_line)*100,'%')
    fx[i], err = integrate.quad(fxy, -10, 15, args=(x_line[i]))

for i in range(0, len(x_line)):
    for j in range(0, len(y_line)):
        fy_giv_x[i,j] = fxy(x_line[i], y_line[j])/fx[i]


fx = fx/np.sum(fx) # normalize
fy_giv_x = fy_giv_x/np.sum(fy_giv_x) # normalize
Fx1 = np.cumsum(fx)
Fx2_giv_x1 = np.cumsum(fy_giv_x, axis=1)

CDF     = lambda x: np.array([np.interp(x[0], x_line, Fx1), \
                              np.interp(x[1], y_line, [np.interp(x[0], x_line, Fx2_giv_x1[:,i]) for i in range(0, len(x_line))])])

# CDF_inv = lambda u: np.array([np.interp(u[0], Fx1, x_line), \
#                               np.interp(u[1], [np.interp(np.interp(u[0], Fx1, x_line), x_line, Fx2_giv_x1[:,i]) for i in range(0, len(x_line))], y_line)])

CDF_inv = lambda u: compute_CDF_inv(u, x_line, y_line, Fx1, Fx2_giv_x1)

def compute_CDF_inv(u, x_line, y_line, Fx1, Fx2_giv_x1):
    x = np.zeros(2)
    x[0] = np.interp(u[0], Fx1, x_line)
    x[1] = np.interp(u[1], [np.interp(x[0], x_line, Fx2_giv_x1[:,i]) for i in range(0, len(x_line))], y_line)
    return x

transform_U2X = lambda u: CDF_inv(phi(u))
transform_X2U = lambda x: phi_inv(CDF(x))

u = transform_U2X([0,0])


# transform seed from X to U
initial_theta_u = transform_X2U(initial_theta)

# apply mcmc-method
theta_u = cs.cond_sampling(initial_theta_u, n_samples, rho_k, burnInPeriod, lagPeriod)

plt.figure()
plt.scatter(theta_u[0,:], theta_u[1,:])

# transform chain from U to X
theta_x = np.zeros(theta_u.shape)
for i in range(0, theta_u.shape[1]):
    theta_x[:,i] = transform_U2X(theta_u[:,i])

plt.figure()
plt.scatter(theta_x[0,:], theta_x[1,:])


# x_new = np.linspace(-5,10,50)
# plt.figure()
# plt.plot(x_new, CDF(x_new))

# x_new_new = np.linspace(0,1,50)
# plt.figure()
# plt.plot(x_new_new, CDF_inv(x_new_new))


plt.show()

# OUTPUT
print('E[X1] =', round(theta_x[0,:].mean(), 5))
print('E[X2] =', round(theta_x[1,:].mean(), 5))

np.save('python/data/samples_2D_pdf_cs.npy', theta_x)

# plot samples
# uplt.plot_hist(theta[0,:], target_PDF, 2)
# uplt.plot_scatter_with_contour(theta, target_PDF)
# uplt.plot_mixing(theta[0, :1000])
# uplt.plot_scatter_with_hist(theta[:, :5000], target_PDF)
# uplt.plot_autocorr(theta[0,:], 50, 1)
# uplt.plot_autocorr(theta[1,:], 50, 2)
# plt.show()
