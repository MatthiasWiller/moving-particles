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
# Version 2017-09
# ---------------------------------------------------------------------------
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 20})
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rcParams['text.usetex'] = True
   
# ---------------------------------------------------------------------------
# STANDARD INPUT
# ---------------------------------------------------------------------------
np.random.seed(1)
savepdf = True

fh = lambda x: 2*np.multiply(np.cos(0.2*np.pi*x),x)

# range
x = np.linspace(-5,5,500)
N = len(x)
# Sampled data points from the generating function
M = 5
selection = np.zeros(N, int)
j = np.random.random_integers(0, N-1, M)
# mark them
selection[j] = 1
x_data = np.array([x[i] for i in range(0,N) if selection[i] == 1])
x_new = np.array([x[i] for i in range(0,N) if selection[i] == 0])
f_data = fh(x_data)
# ---------------------------------------------------------------------------
# DEFINE FUNCTIONS
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# GAUSSIAN PROCESS REGRESSION (GPR)
# ---------------------------------------------------------------------------
x = np.concatenate((x_data, x_new), axis=0)
theta2 = .2 # correlation length [0.2, 2]
sigma2 = 20  # sigma2 in kernel [2, 20]
K = np.zeros((N,N))
# computing the interpolation using all x's
# It is expected that for points used to build the GP cov. matrix, the
# uncertainty is reduced...
for i in range(0, N):
    for j in range(0, N):
        K[i,j] = sigma2 * np.exp(-0.5*(x[i]-x[j])**2/theta2)

# upper left corner of K
Kaa = K[:M, :M]
# lower right corner of K
Kbb = K[M:, M:]
# upper right corner of K
Kab = K[:M, M:]
# mean of posterior
m = np.inner(np.inner(Kab.T,np.linalg.inv(Kaa)),f_data.T)
# cov. matrix of posterior
D = Kbb - np.inner(np.inner(Kab.T,np.linalg.inv(Kaa)),Kab.T)


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------


plt.figure()
plt.plot(x_data, f_data, 'o', color='k')
x = np.sort(x)
plt.plot(x, fh(x), '--', color='green')
plt.plot(x_new, m, '-', color='navy')
plt.fill_between(x_new, m-np.sqrt(np.diagonal(D)), m+np.sqrt(np.diagonal(D)), color='lightgray')
plt.plot()

# axis
plt.xlabel(r'$x$')
plt.ylabel(r'$f(x)$')


plt.tight_layout()
if savepdf:
    plt.savefig('gpr_1D.pdf', format='pdf', dpi=50, bbox_inches='tight')

plt.show()