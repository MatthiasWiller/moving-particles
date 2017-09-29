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
c = 0 # correlation rho = 0
# c = 1 # correlation rho = -0.40
target_PDF = lambda x: (1 - c*(1-x[0]-x[1]) + c*c*x[0]*x[1])*np.exp(-(x[0]+x[1]+c*x[0]*x[1]))

# get grid, minimum, maximum
x       = np.linspace(0, 5, 100)
X, Y    = np.meshgrid(x, x)
Z       = target_PDF([X, Y])

plt.figure()

plt.contour(X,Y,Z)

fig = plt.figure()

ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=20, cstride=20, cmap=cm.pink_r, antialiased=False, alpha=1.0)
ax.plot_wireframe(X, Y, Z, rstride=20, cstride=20, linewidth=0.5, color='black', alpha=1.0)

plt.show()