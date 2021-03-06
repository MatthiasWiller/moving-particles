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

example = 3
savepdf = True
# savepdf = True

# load data
direction = 'python/data/'

linestyle = ['yH-','ro-','bs-']

# -------------------------------------------------------------------------------------------
# EXAMPLE 1
# -------------------------------------------------------------------------------------------
if example == 1:
    # load samples
    theta_list_list  = np.load(direction + 'mp_example_1_d2_N100_Nsim1_b5_cs_sss2_theta_list.npy')
    theta_list = theta_list_list[0]
    sample_id_list = [2,5,8]


    # beta = 5.1993       # for pf = 10^-7
    # beta = 4.7534       # for pf = 10^-6 (N=5*1e9)
    # beta = 4.2649       # for pf = 10^-5 (N=5*1e8)
    # beta = 3.7190       # for pf = 10^-4 (N=5*1e7)
    beta = 3.0902       # for pf = 10^-3 (N=5*1e6)
    # beta = 2.3263       # for pf = 10^-2 (N=5*1e5)
    d = 2
    LSF  = lambda u, beta: u.sum(axis=0)/np.sqrt(d) + beta


    # get grid, minimum, maximum
    x       = np.linspace(-6, 6, 300)
    X, Y    = np.meshgrid(x, x)
    Z       = LSF(np.array([X, Y]), beta)

    min_x = min(X.flatten())
    min_y = min(Y.flatten())
    max_x = max(X.flatten())
    max_y = max(Y.flatten())

    # PLOT 3D-plot
    fig = plt.figure()

    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=20, cstride=20, cmap=cm.pink_r, antialiased=False, alpha=1.0)
    ax.plot_wireframe(X, Y, Z, rstride=20, cstride=20, linewidth=0.5, color='black', alpha=1.0)
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
        # plt.savefig('example'+ repr(example) +'_lsf_3D.pdf', format='pdf', dpi=50, bbox_inches='tight')
        # plt.savefig('density.pdf', format='pdf', dpi=50)
        print('no pdf saved!')


    # PLOT 2D-plot
    fig = plt.figure()
    plt.axes().set_aspect('equal')

    plt.contour(X, Y, Z, [0], colors='k')

    for i in range(0, len(sample_id_list)):
        sample = sample_id_list[i]
        theta = np.array(theta_list[sample])
        plt.plot(theta[:, 0], theta[:, 1], linestyle[i])

    # set labels
    plt.xlabel(r'$u_1$')
    plt.ylabel(r'$u_2$')

    plt.xlim(-6, 6)
    plt.ylim(-6, 6)

    plt.xticks([-5, 0, 5])
    plt.yticks([-5, 0, 5])
    
    plt.tight_layout()
    if savepdf:
        plt.savefig('example'+ repr(example) +'_lsf_w_samples_2D.pdf', format='pdf', dpi=50, bbox_inches='tight')

    plt.show()

# -------------------------------------------------------------------------------------------
# EXAMPLE 2 (liebscher)
# -------------------------------------------------------------------------------------------
elif example == 2: 
    # load samples
    theta_list_list  = np.load(direction + 'mp_liebscher_N100_Nsim2_b20_cs_sss2_theta_list.npy')
    theta_list = theta_list_list[0]
    sample_id_list = [4,5,1]


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
    ax.plot_surface(X, Y, Z, rstride=20, cstride=20, cmap=cm.pink_r, antialiased=False, alpha=1.0)
    ax.plot_wireframe(X, Y, Z, rstride=20, cstride=20, linewidth=0.5, color='black', alpha=1.0)
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
        # plt.savefig('example'+ repr(example) +'_lsf_3D.pdf', format='pdf', dpi=50, bbox_inches='tight')
        # plt.savefig('density.pdf', format='pdf', dpi=50)
        print('no pdf saved!')

    # PLOT 2D-plot
    fig = plt.figure()
    plt.axes().set_aspect('equal')

    # plt.contour(X, Y, Z, [7.5], cmap=cm.jet)
    plt.contour(X, Y, Z, [7.5], colors='k')

    for i in range(0, len(sample_id_list)):
        sample = sample_id_list[i]
        theta = np.array(theta_list[sample])
        plt.plot(theta[:, 0], theta[:, 1], linestyle[i])


    # set labels
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')

    plt.xlim(-2, 6)
    plt.ylim(-2, 6)

    plt.xticks([-2, 0, 2, 4, 6])
    plt.yticks([-2, 0, 2, 4, 6])

    plt.tight_layout()
    if savepdf:
        plt.savefig('example'+ repr(example) +'_lsf_w_samples_2D.pdf', format='pdf', dpi=50, bbox_inches='tight')

    plt.show()



# -------------------------------------------------------------------------------------------
# EXAMPLE 3 (waarts)
# -------------------------------------------------------------------------------------------
elif example == 3:
    # load samples
    theta_list_list  = np.load(direction + 'mp_waarts_N100_Nsim2_b20_cs_sss2_theta_list.npy')
    theta_list = theta_list_list[0]
    sample_id_list = [7,6,3]


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
    ax.plot_surface(X, Y, Z, rstride=20, cstride=20, cmap=cm.pink_r, antialiased=False, alpha=1.0)
    ax.plot_wireframe(X, Y, Z, rstride=20, cstride=20, linewidth=0.5, color='black', alpha=1.0)
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
        # plt.savefig('example'+ repr(example) +'_lsf_3D.pdf', format='pdf', dpi=50, bbox_inches='tight')
        # plt.savefig('density.pdf', format='pdf', dpi=50)
        print('no pdf saved!')

    # PLOT 2D-plot
    fig = plt.figure()
    plt.axes().set_aspect('equal')

    plt.contour(X, Y, Z, [0], linewidth=.2, colors='k')

    for i in range(0, len(sample_id_list)):
        sample = sample_id_list[i]
        theta = np.array(theta_list[sample])
        plt.plot(theta[:, 0], theta[:, 1], linestyle[i])

    # set labels
    plt.xlabel(r'$u_1$')
    plt.ylabel(r'$u_2$')

    plt.xlim(-7, 7)
    plt.ylim(-7, 7)

    plt.xticks([-5, 0, 5])
    plt.yticks([-5, 0, 5])

    plt.tight_layout()
    if savepdf:
        plt.savefig('example'+ repr(example) +'_lsf_w_samples_2D.pdf', format='pdf', dpi=50, bbox_inches='tight')


    fig = plt.figure()

    for i in range(0, len(sample_id_list)):
        sample = sample_id_list[i]
        theta = np.array(theta_list[sample])
        m = theta.shape[0]
        g = np.zeros(m)
        for j in range(0,m):
            g[j] = LSF(theta[j, :])

        plt.plot(g, linestyle[i])


        

    plt.xlabel(r'$m$')
    plt.ylabel(r'Limit state function value, $g(X)$')

    plt.tight_layout()
    if savepdf:
        plt.savefig('example' + repr(example) + '_lsf_over_m.pdf', format='pdf', dpi=50, bbox_inches='tight')
    
    plt.show()




# -------------------------------------------------------------------------------------------
# EXAMPLE 4 (au and beck)
# -------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------
# EXAMPLE 5 (breitung)
# -------------------------------------------------------------------------------------------elif example == 5:
if example == 5:
    LSF = lambda x: np.minimum(5-x[0], 4+x[1])
    # LSF = lambda x: np.minimum(5-x[0], 1/(1+np.exp(-2*(x[1]+4)))-0.5)

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
    ax.plot_surface(X, Y, Z, rstride=20, cstride=20, cmap=cm.pink_r, antialiased=False, alpha=1.0)
    ax.plot_wireframe(X, Y, Z, rstride=20, cstride=20, linewidth=0.5, color='black', alpha=1.0)
    ax.view_init(elev=60, azim=-120)

    # axes and title config
    ax.set_xlabel('$u_1$', labelpad=15)
    ax.xaxis.set_rotate_label(False) # disable automatic rotation
    ax.set_ylabel('$u_2$', rotation = 0, labelpad=15)
    ax.yaxis.set_rotate_label(False)
    ax.set_zlabel('$g(u_1, u_2)$',rotation=93, labelpad=7)
    ax.zaxis.set_rotate_label(False)
    ax.set_xlim3d(min_x, max_x)
    ax.set_ylim3d(min_y, max_y)
    ax.set_zlim3d(-3, 13)
    ax.set_zticks([0, 5, 10])

    plt.tight_layout()
    if savepdf:
        plt.savefig('example'+ repr(example) +'_lsf_3D.pdf', format='pdf', dpi=50, bbox_inches='tight')
        # plt.savefig('density.pdf', format='pdf', dpi=50)


    # PLOT 2D-plot
    fig = plt.figure()
    plt.axes().set_aspect('equal')

    plt.contour(X, Y, Z, [0], linewidth=.2, colors='k')

    # Z0 = gaussian(X,Y)
    # CS0 = plt.contour(X, Y, Z0, [1e-5, 1e-3, 2e-2, 1e-1], linewidths=.2, colors='k')

    # set labels
    plt.xlabel(r'$u_1$')
    plt.ylabel(r'$u_2$')

    plt.xlim(-7, 7)
    plt.ylim(-7, 7)

    plt.xticks([-5, 0, 5])
    plt.yticks([-5, 0, 5])

    plt.tight_layout()
    if savepdf:
        plt.savefig('example'+ repr(example) +'_lsf_2D.pdf', format='pdf', dpi=50, bbox_inches='tight')

    plt.show()

