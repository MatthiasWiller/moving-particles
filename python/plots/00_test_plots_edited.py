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

Z = np.round(Z,4)

min_x = min(X.flatten())
min_y = min(Y.flatten())
min_z = min(Z.flatten())

max_x = max(X.flatten())
max_y = max(Y.flatten())
max_z = max(Z.flatten())


# plotting
fig  = plt.figure(1)
ax   = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=5, cstride=10, cmap=cm.pink_r, linewidth=5, alpha=1.0, antialiased=False)
ax.plot_wireframe(X, Y, Z, rstride=5, cstride=10, linewidth=0.5, color='black')  
# axes and title config 
ax.view_init(elev=45, azim=45)
ax.set_xlabel('$x_1$', labelpad=15)
ax.xaxis.set_rotate_label(False) # disable automatic rotation
ax.set_ylabel('$x_2$', labelpad=15)
ax.yaxis.set_rotate_label(False) # disable automatic rotation
ax.set_zlabel('$f(x_1, x_2)$', rotation=90, labelpad=15)
ax.zaxis.set_rotate_label(False)
ax.set_xlim3d(min_x, max_x)
ax.set_ylim3d(min_y, max_y)
ax.set_zlim3d(min_z, 0.25)

plt.tight_layout()
plt.savefig('density.pdf', format='pdf', dpi=50, bbox_inches='tight')


# fig  = plt.figure(2)
# pcol = plt.contourf(X, Y, Z, 10, cmap=cm.BuGn, extend='both')#, vmin=C_FF.min(), vmax=C_FF.max())
# plt.contour(X, Y, Z, pcol.levels, colors='k')
# #plt.clabel(pcol, inline=1, fontsize=10)
# cb = plt.colorbar(pcol, shrink=0.95, aspect=15, pad = 0.05)
# cb.formatter.set_powerlimits((0, 0))
# cb.ax.yaxis.set_offset_position('left')                         
# cb.update_ticks()
# ax.set_xlabel('$x_1$', labelpad=15)
# ax.set_ylabel('$x_2$', rotation = 0, labelpad=15)
# plt.xlim(min_x, max_x)
# plt.ylim(min_y, max_y)
# plt.gca().set_aspect('equal', adjustable='box')

# plt.tight_layout()
# plt.savefig('density2.pdf', format='pdf', dpi=50, bbox_inches='tight')
