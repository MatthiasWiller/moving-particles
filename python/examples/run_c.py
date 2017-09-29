"""
Author: Matthias Willer 2017
"""

import time as timer

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scps
import scipy.integrate as integrate
import scipy.interpolate as interpolate

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker    


np.random.seed(0)

# target pdf 
mu1 = np.array([1, 1])
rho1 = 0.6
sig1 = np.array([np.array([1**2, rho1*1*2 ]),np.array([rho1*1*2,2**2])])
mu2 = np.array([3, 1.5])
rho2 = -0.5
sig2 = np.array([np.array([0.7**2,rho2*0.7*0.3]),np.array([rho2*0.7*0.3,0.3**2])])

target_PDF = lambda x: scps.multivariate_normal.pdf(x, mu1, sig1) \
                     + scps.multivariate_normal.pdf(x, mu2, sig2)

# get grid, minimum, maximum
n = 100
x       = np.linspace(-5, 5, n)
X, Y    = np.meshgrid(x, x)
Z = np.zeros((n, n))
for i in range(0, len(X)):
    for j in range(0, len(Y)):
        Z[i,j] = target_PDF(np.array([X[i,j], Y[i,j]]))

plt.figure()

plt.contour(X,Y,Z, [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

fig = plt.figure()

ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=4, cstride=4, cmap=cm.pink_r, antialiased=False, alpha=1.0)
ax.plot_wireframe(X, Y, Z, rstride=4, cstride=4, linewidth=0.5, color='black', alpha=1.0)

plt.show()