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
# Version 2017-10
# ---------------------------------------------------------------------------
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 23})
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rcParams['text.usetex'] = True

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker     

# INPUT 

# target pdf 2D (Example from 05_2D_pdf.py)
def target_PDF_2D(x):
    c = 1/20216.335877
    # f(x,y) = 1/20216.335877 * exp(-(x*x*y*y + x*x + y*y - 8*x - 8*y)/2)
    return c * np.exp(-(x[0]*x[0]*x[1]*x[1] + x[0]*x[0] + x[1]*x[1] - 8*x[0] - 8*x[1])/2)


# get grid, minimum, maximum
x       = np.linspace(-1, 7, 300)
X, Y    = np.meshgrid(x, x)
Z       = target_PDF_2D([X, Y])

min_x = min(X.flatten())
min_y = min(Y.flatten())
min_z = min(Z.flatten())

max_x = max(X.flatten())
max_y = max(Y.flatten())
max_z = max(Z.flatten())


# plotting
fig = plt.figure()

ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=5, cstride=10, cmap=cm.pink_r, linewidth=5, alpha=1.0, antialiased=False)
ax.plot_wireframe(X, Y, Z, rstride=5, cstride=10, linewidth=0.5, color='black')  
ax.view_init(elev=45, azim=55)

# axes and title config

ax.set_xlabel('$x_1$', labelpad=15)
ax.xaxis.set_rotate_label(False) # disable automatic rotation
ax.set_ylabel('$x_2$', rotation = 0, labelpad=15)
ax.yaxis.set_rotate_label(False)
ax.set_zlabel('$f(x_1, x_2)$',rotation=93, labelpad=7)
ax.zaxis.set_rotate_label(False)
ax.set_xlim3d(min_x, max_x)
ax.set_ylim3d(min_y, max_y)
ax.set_zlim3d(min_z, 0.25)

ax.set_zticks([0, 0.1, 0.2])

plt.tight_layout()
plt.savefig('density.pdf', format='pdf', dpi=50, bbox_inches='tight')
# plt.savefig('density.pdf', format='pdf', dpi=50)


plt.show()
