"""
# ---------------------------------------------------------------------------
# Test function to test plots
# ---------------------------------------------------------------------------
# Created by:
# Matthias Willer (matthias.willer@tum.de)
# Engineering Risk Analysis Group
# Technische Universitat Munchen
# www.era.bgu.tum.de
# ---------------------------------------------------------------------------
# Version 2017-05
# ---------------------------------------------------------------------------
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rcParams['text.usetex'] = True
   
savepdf = True

# DEFINE FUNCTIONS

# Monte Carlo
monte_carlo = lambda pf, delta: (1-pf)/(pf*delta**2)

# Subset Simulation
p0 = 0.1
subset = lambda pf, delta, gamma: ((np.log(pf)/np.log(p0)*(1-p0)/np.sqrt(p0))**2 + np.log(pf)/np.log(p0)*(1-p0)/p0)*((1+gamma)/(delta**2))

# Moving particles
moving_particles = lambda pf, delta, Nb: (Nb*(-np.log(pf))+1)*(-np.log(pf))/(np.log(delta**2 + 1))

# parameters
# pf_line = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
pf_line = np.logspace(-12, 0, 1000)

delta_line = np.linspace(0.001, 0.8, 100)

X, Y = np.meshgrid(pf_line, delta_line)

# -----------------------------------------------------------------------------
# COST over PROBABILITY OF FAILURE
# -----------------------------------------------------------------------------

# initialization
nelements = len(pf_line)

cost_mc  = monte_carlo(X,Y)
cost_sus1 = subset(X,Y, 1.5)
cost_sus2 = subset(X,Y, 2.5)
cost_mp  = moving_particles(X,Y, 5)


# plotting
fig  = plt.figure()

plt.contour(X, Y, cost_mc, [1e3, 1e4, 1e5, 1e6], colors='k')
plt.contour(X, Y, cost_sus1, [1e3, 1e4, 1e5, 1e6], colors='b')
plt.contour(X, Y, cost_sus2, [1e3, 1e4, 1e5, 1e6], colors='m')
plt.contour(X, Y, cost_mp, [1e3, 1e4, 1e5, 1e6], label=r'MP', colors='r')

# legend
# plt.legend(loc='upper left')

# xaxis
plt.xscale('log')
plt.xlim(1e-12, 0.99)
plt.gca().invert_xaxis()
plt.xlabel(r'Probability of failure, $p_f$')

# yaxis
# plt.yscale('log')
plt.ylim(0, 0.8)
plt.ylabel(r'Coefficient of variation, $\delta_{p_f}$')

plt.tight_layout()
if savepdf:
    plt.savefig('comp_cost.pdf', format='pdf', dpi=50, bbox_inches='tight')

plt.show()