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
import scipy.stats as scps
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 23})
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rcParams['text.usetex'] = True

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker     



# INPUT 

example = 1
savepdf = False


gaussian = lambda x, y: (1/(2*np.pi) * np.exp(-(x**2/2 + y**2/2)))

# -------------------------------------------------------------------------------------------
# EXAMPLE 1
# -------------------------------------------------------------------------------------------
if example == 1:
    #beta = 5.1993       # for pf = 10^-7
    beta4 = 4.7534       # for pf = 10^-6 (N=5*1e9)
    beta3 = 4.2649       # for pf = 10^-5 (N=5*1e8)
    beta2 = 3.7190       # for pf = 10^-4 (N=5*1e7)
    #beta = 3.0902       # for pf = 10^-3 (N=5*1e6)
    beta1 = 2.3263       # for pf = 10^-2 (N=5*1e5)
    d = 2
    LSF  = lambda u, beta: u.sum(axis=0)/np.sqrt(d) + beta


    # get grid, minimum, maximum
    x       = np.linspace(-6, 6, 300)
    X, Y    = np.meshgrid(x, x)
    Z1       = LSF(np.array([X, Y]), beta1)
    Z2       = LSF(np.array([X, Y]), beta2)
    Z3       = LSF(np.array([X, Y]), beta3)
    Z4       = LSF(np.array([X, Y]), beta4)

    min_x = min(X.flatten())
    min_y = min(Y.flatten())
    max_x = max(X.flatten())
    max_y = max(Y.flatten())

    # PLOT 3D-plot
    fig = plt.figure()

    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z1, rstride=6, cstride=6, cmap=cm.pink_r, antialiased=False, alpha=1.0)
    ax.plot_wireframe(X, Y, Z1, rstride=6, cstride=6, linewidth=0.5, color='black', alpha=1.0)
    ax.view_init(elev=42, azim=-40)

    # axes and title config
    ax.set_xlabel('$u_1$', labelpad=15)
    ax.xaxis.set_rotate_label(False) # disable automatic rotation
    ax.set_ylabel('$u_2$', rotation = 0, labelpad=15)
    ax.yaxis.set_rotate_label(False)
    ax.set_zlabel('$g(u_1, u_2)$',rotation=93, labelpad=7)
    ax.zaxis.set_rotate_label(False)
    ax.set_xlim3d(min_x, max_x)
    ax.set_ylim3d(min_y, max_y)
    # ax.set_zlim3d(min_z, 0.25)
    # ax.set_zticks([0, 0.1, 0.2])
    ax.set_xticks([-5, 0, 5])
    ax.set_yticks([-5, 0, 5])

    plt.tight_layout()
    if savepdf:
        plt.savefig('example'+ repr(example) +'_lsf_3D.pdf', format='pdf', dpi=50, bbox_inches='tight')
        # plt.savefig('density.pdf', format='pdf', dpi=50)


    # PLOT 2D-plot
    fig = plt.figure()
    plt.axes().set_aspect('equal')

    Z0 = gaussian(X,Y)
    CS0 = plt.contour(X, Y, Z0, [1e-5, 1e-3, 2e-2, 1e-1], linewidths=.2, colors='k')

    # CS1 = plt.contour(X, Y, Z1, [0], colors='k')
    # plt.clabel(CS1, fontsize=9, inline=1, fmt=r'$p_f = 10^{-2}$', manual=[(0, 0)])

    # CS2 = plt.contour(X, Y, Z2, [0], colors='k')
    # plt.clabel(CS2, fontsize=9, inline=1, fmt=r'$p_f = 10^{-4}$', manual=[(0, 0)])

    # CS4 = plt.contour(X, Y, Z4, [0], colors='k')
    # plt.clabel(CS4, fontsize=9, inline=1, fmt=r'$p_f = 10^{-6}$', manual=[(0, 0)])


    CS1 = plt.contour(X, Y, Z1, [0], colors='k')
    plt.clabel(CS1, fontsize=15, inline=3, inline_spacing=5, fmt=r'$\beta = 2.33$')

    CS2 = plt.contour(X, Y, Z2, [0], colors='k')
    plt.clabel(CS2, fontsize=9, inline=1, inline_spacing=5, fmt=r'$\beta = 3.72$', manual=[(0, 0)])

    CS4 = plt.contour(X, Y, Z4, [0], colors='k')
    plt.clabel(CS4, fontsize=9, inline=1, inline_spacing=5, fmt=r'$\beta = 4.75$', manual=[(0, 0)])

    # set labels
    plt.xlabel(r'$u_1$')
    plt.ylabel(r'$u_2$')

    plt.xlim(-6, 6)
    plt.ylim(-6, 6)

    plt.xticks([-5, 0, 5])
    plt.yticks([-5, 0, 5])

    
    plt.tight_layout()
    if savepdf:
        plt.savefig('example'+ repr(example) +'_lsf_2D.pdf', format='pdf', dpi=50, bbox_inches='tight')

    plt.show()

# -------------------------------------------------------------------------------------------
# EXAMPLE 2 (liebscher)
# -------------------------------------------------------------------------------------------
elif example == 2: 
    LSF   = lambda x: 8* np.exp(-(x[0]**2 + x[1]**2)) + 2* np.exp(-((x[0]-5)**2 + (x[1]-4)**2)) + 1 + x[0]*x[1]/10

    # get grid, minimum, maximum
    x       = np.linspace(-2, 6, 300)
    X, Y    = np.meshgrid(x, x)
    Z       = LSF([X, Y])

    min_x = min(X.flatten())
    min_y = min(Y.flatten())
    min_z = min(Z.flatten())
    max_x = max(X.flatten())
    max_y = max(Y.flatten())
    max_z = max(Z.flatten())

    # PLOT 3D-plot
    fig = plt.figure()

    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=6, cstride=6, cmap=cm.pink_r, antialiased=False, alpha=1.0)
    ax.plot_wireframe(X, Y, Z, rstride=6, cstride=6, linewidth=0.5, color='black', alpha=1.0)
    ax.view_init(elev=42, azim=-40)

    # axes and title config
    ax.set_xlabel('$x_1$', labelpad=15)
    ax.xaxis.set_rotate_label(False) # disable automatic rotation
    ax.set_ylabel('$x_2$', rotation = 0, labelpad=15)
    ax.yaxis.set_rotate_label(False)
    ax.set_zlabel('$z(x_1, x_2)$',rotation=93, labelpad=7)
    ax.zaxis.set_rotate_label(False)
    ax.set_xlim3d(min_x, max_x)
    ax.set_ylim3d(min_y, max_y)
    # ax.set_zlim3d(min_z, 0.25)
    # ax.set_zticks([0, 0.1, 0.2])

    plt.tight_layout()
    if savepdf:
        plt.savefig('example'+ repr(example) +'_lsf_3D.pdf', format='pdf', dpi=50, bbox_inches='tight')
        # plt.savefig('density.pdf', format='pdf', dpi=50)


    # PLOT 2D-plot
    fig = plt.figure()
    plt.contour(X, Y, Z, [7.5], cmap=cm.jet)

    # set labels
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')

    plt.xlim(-2, 6)
    plt.ylim(-2, 6)

    plt.xticks([-2, 0, 2, 4, 6])
    plt.yticks([-2, 0, 2, 4, 6])

    plt.tight_layout()
    if savepdf:
        plt.savefig('example'+ repr(example) +'_lsf_2D.pdf', format='pdf', dpi=50, bbox_inches='tight')

    plt.show()



# -------------------------------------------------------------------------------------------
# EXAMPLE 3 (waarts)
# -------------------------------------------------------------------------------------------
elif example == 3:
    LSF = lambda u: np.minimum(3 + 0.1*(u[0] - u[1])**2 - 2**(-0.5) * np.absolute(u[0] + u[1]), 7* 2**(-0.5) - np.absolute(u[0] - u[1]))
    
    # get grid, minimum, maximum
    x       = np.linspace(-7, 7, 300)
    X, Y    = np.meshgrid(x, x)
    Z       = LSF([X, Y])

    min_x = min(X.flatten())
    min_y = min(Y.flatten())
    min_z = min(Z.flatten())
    max_x = max(X.flatten())
    max_y = max(Y.flatten())
    max_z = max(Z.flatten())

    # PLOT 3D-plot
    fig = plt.figure()

    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=6, cstride=6, cmap=cm.pink_r, antialiased=False, alpha=1.0)
    ax.plot_wireframe(X, Y, Z, rstride=6, cstride=6, linewidth=0.5, color='black', alpha=1.0)
    ax.view_init(elev=42, azim=-135)

    # axes and title config
    ax.set_xlabel('$u_1$', labelpad=15)
    ax.xaxis.set_rotate_label(False) # disable automatic rotation
    ax.set_ylabel('$u_2$', rotation = 0, labelpad=15)
    ax.yaxis.set_rotate_label(False)
    ax.set_zlabel('$g(u_1, u_2)$',rotation=93, labelpad=7)
    ax.zaxis.set_rotate_label(False)
    ax.set_xlim3d(min_x, max_x)
    ax.set_ylim3d(min_y, max_y)
    ax.set_zlim3d(-8.5, 6)
    ax.set_zticks([5, 0, -5])

    plt.tight_layout()
    if savepdf:
        plt.savefig('example'+ repr(example) +'_lsf_3D.pdf', format='pdf', dpi=50, bbox_inches='tight')
        # plt.savefig('density.pdf', format='pdf', dpi=50)


    # PLOT 2D-plot
    fig = plt.figure()

    plt.contour(X, Y, Z, [0], cmap=cm.jet, linewidth=.5, colors='k')

    Z0 = gaussian(X,Y)
    CS0 = plt.contour(X, Y, Z0, [1e-5, 1e-3, 2e-2, 1e-1], linewidths=.2, colors='k')

    # set labels
    plt.xlabel(r'$u_1$')
    plt.ylabel(r'$u_2$')

    plt.xlim(-7, 7)
    plt.ylim(-7, 7)

    plt.xticks([-5, 0, 5])
    plt.yticks([-5, 0, 5])

    plt.axes().set_aspect('equal')

    plt.tight_layout()
    if savepdf:
        plt.savefig('example'+ repr(example) +'_lsf_2D.pdf', format='pdf', dpi=50, bbox_inches='tight')

    plt.show()




# -------------------------------------------------------------------------------------------
# EXAMPLE 4 (au and beck)
# -------------------------------------------------------------------------------------------

# example 5
elif example == 5:
    LSF = lambda x: x



