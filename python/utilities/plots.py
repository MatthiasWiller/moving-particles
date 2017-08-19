"""
# ---------------------------------------------------------------------------
# Several plotting functions
# ---------------------------------------------------------------------------
# Created by:
# Matthias Willer (matthias.willer@tum.de)
# Engineering Risk Analysis Group
# Technische Universitat Munchen
# www.era.bgu.tum.de
# ---------------------------------------------------------------------------
# Version 2017-07
# ---------------------------------------------------------------------------
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.pyplot import cm
from matplotlib import rcParams
from matplotlib import ticker
from matplotlib.ticker import NullFormatter

from mpl_toolkits.mplot3d import Axes3D

# create figure object with LaTeX font
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = 22
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


# -----------------------------------------------------------------------------------------
# histogram plot
def plot_hist(x, target_PDF=0, dimension=1):

    len_x       = len(x)
    n           = np.sqrt(len_x)
    num_bins    = np.math.ceil(n)

    # the histogram of the data
    plt.figure()
    n, bins, patches = plt.hist(x, num_bins, normed=1, color='navy')

    # add a 'best fit' line
    if((target_PDF != 0) and (dimension == 1)):
        # for 1D case
        x_bins = np.arange(-10,20,0.05)
        y = target_PDF(x_bins)
        plt.plot(x_bins, y, '-', color='red')

    if((target_PDF != 0) and (dimension == 2)):
        # for 2D case
        y = compute_marginal_PDF(target_PDF, bins, 0)
        plt.plot(bins, y, '-', color='red')

    # plt.ylim(0,1)
    # plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])


    # set labels
    plt.xlabel(r'$x$')
    plt.ylabel(r'Frequency $p$')
    plt.tight_layout()
    # plt.savefig('plot_hist.pdf', format='pdf', dpi=50, bbox_inches='tight')


# -----------------------------------------------------------------------------------------
# plot values of index
def plot_mixing(x):

    n_samples = len(x)

    plt.figure()
    plt.plot(x, color='navy')

    # set labels
    plt.xlabel(r'Number of samples, $n$')
    plt.ylabel(r'$x$')
    plt.xlim(0, n_samples)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.tight_layout()
    #plt.savefig('plot_mixing.pdf', format='pdf', dpi=50, bbox_inches='tight')


# -----------------------------------------------------------------------------------------
# plot of the estimated autocorrelation of samples
def plot_autocorr(x, lag, id=0):

    # compute sample autocorrelation
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
    plt.plot(rho, '.')


    # set labels
    plt.xlabel(r'Lag, $k$')
    plt.ylabel(r'$\hat{R}(k)')
    plt.xlim(0, lag)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.tight_layout()
    plt.savefig('plot_autocorr_'+str(id)+'.pdf', format='pdf', dpi=50, bbox_inches='tight')


# -----------------------------------------------------------------------------------------
# make a nice scatter plot with contour lines of the pdf and samples    
def plot_scatter_with_contour(theta, target_PDF):

    x       = np.linspace(-2, 8, 100)
    X, Y    = np.meshgrid(x, x)
    Z       = target_PDF([X, Y])

    # plot results
    plt.figure()
    plt.contour(X, Y, Z, 7, cmap=cm.jet)
    #plt.scatter(theta[0,:], theta[1,:], marker='o', facecolors='None', color='navy', linewidths=1, label='Circles')
    plt.scatter(theta[0,:], theta[1,:], s=1, color='blue', marker='o', linestyle='None')

    #ax.plot_surface(theta[0,:], theta[1,:], target_PDF(theta) )

    # set labels
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.xticks([-2, 0, 2, 4, 6, 8])
    plt.yticks([-2, 0, 2, 4, 6, 8])

    plt.gca().set_aspect('equal', adjustable='box')
    #plt.tight_layout()
    #plt.savefig('plot_scatter_with_contour.pdf', format='pdf', dpi=50, bbox_inches='tight')


# -----------------------------------------------------------------------------------------
# plot surface with samples
def plot_surface_with_samples(theta, g, f, g_max_global):
    x       = np.linspace(-2, 6, 100)
    X, Y    = np.meshgrid(x, x)
    Z       = f([X, Y])

    z_plane = 7.5 * np.ones(len(x))

    z_samples = f([theta[:, 0], theta[:, 1]])

    # 3D Plot
    min_x = min(X.flatten())
    min_y = min(Y.flatten())
    min_z = min(Z.flatten())

    max_x = max(X.flatten())
    max_y = max(Y.flatten())
    max_z = max(Z.flatten())

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # plot z-surface
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, cmap=cm.pink_r, antialiased=False, alpha=1.0)
    ax.plot_wireframe(X, Y, Z, rstride=3, cstride=3, linewidth=0.5, color='black', alpha=1.0)

    # custom color map
    theCM = cm.get_cmap()
    theCM._init()
    alphas = np.abs(np.linspace(-1.0, 1.0, theCM.N))
    theCM._lut[:-3, -1] = alphas

    # plot limit-surface (= 7.5)
    ax.plot_surface(X, Y, z_plane, rstride=5, cstride=5, cmap=theCM, antialiased=False, alpha=0.4)

    # get colormap
    colors = []


    my_cmap = cm.get_cmap('viridis')
    for i in range(0, np.size(theta, axis=0)):
        colors.append( my_cmap(1.0 - g[i]/g_max_global) )  # color will now be an RGBA tuple

    for i in range(0, np.size(theta, axis=0)):
        # use ax.plot instead of ax.scatter to display the samples in front of the surface
        ax.plot([theta[i, 0]], [theta[i, 1]], [z_samples[i]],\
                linestyle='none', marker='+', mfc='none', markeredgecolor=colors[i])

    # set view
    ax.view_init(elev=42, azim=-40)

    # axes and title config
    ax.set_xlabel('$x_1$', labelpad=15)
    ax.yaxis.set_rotate_label(False) # disable automatic rotation
    ax.set_ylabel('$x_2$', rotation=15, labelpad=15)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('$z(x)$', rotation=93, labelpad=3)
    ax.set_xlim3d(min_x, max_x)
    ax.set_ylim3d(min_y, max_y)
    ax.set_zlim3d(min_z, max_z)
    ttl = ax.set_title('Benchmark Study')
    ttl.set_position([.5, 0.95])

    plt.tight_layout()


# -----------------------------------------------------------------------------------------
# plot the surface of a 2D pdf in 3D
def plot_surface_custom(target_PDF):
    x       = np.linspace(-2, 8, 100)
    X, Y    = np.meshgrid(x, x)
    Z       = target_PDF([X, Y])

    plt.figure()
    ax = plt.gca(projection='3d')

    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.pink)
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, cmap=cm.pink_r, antialiased=False, alpha=1.0)
    # # ax.plot_surface(X, Y, Z, rstride=3, cstride=3, cmap=cm.pink_r, antialiased=True)

    ax.plot_wireframe(X, Y, Z, rstride=3, cstride=3, linewidth=0.5, color='black', alpha=1.0)

    # ttl = ax.set_title('Target distribution')
    # ttl.set_position([.5, 0.95])
    ax.view_init(elev=45, azim=40)
    ax.set_xlim3d(-2, 8)
    ax.set_ylim3d(-2, 8)
    ax.set_zlim3d(0, .6)
    ax.set_xlabel(r'x_1')
    ax.set_ylabel(r'x_2')
    ax.set_zlabel(r'f(x)')
    ax.xaxis._axinfo['label']['space_factor'] = 40
    ax.yaxis._axinfo['label']['space_factor'] = 40
    ax.zaxis._axinfo['label']['space_factor'] = 40

    plt.tight_layout()
    plt.savefig('plot3d.pdf', format='pdf', dpi=50, bbox_inches='tight')


# -----------------------------------------------------------------------------------------
# plot combination of Scatter plot with histogram
def plot_scatter_with_hist(x, target_PDF=0):

    nullfmt = NullFormatter()         # no labels

    xx       = np.linspace(-2, 8, 100)
    X, Y    = np.meshgrid(xx, xx)
    Z       = target_PDF([X, Y])


    # definitions for the axes
    left, width         = 0.1, 0.65
    bottom, height      = 0.1, 0.65
    bottom_h = left_h   = left + width + 0.02

    # make rectangulars
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure( figsize=(8, 8) )

    # set up scatter and histograms
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the contour
    axScatter.contour(X, Y, Z, 5, cmap=cm.jet)

    # the scatter plot
    axScatter.scatter(x[0, :], x[1, :], marker='o', facecolors='None', \
                      color='navy', linewidths=1, label='Circles')

    # now determine nice limits by hand:
    binwidth = 0.10

    # choose limits of the plot
    lowerlim = -1
    upperlim = 7

    # set limits of scatter plot
    axScatter.set_xlim((lowerlim, upperlim))
    axScatter.set_ylim((lowerlim, upperlim))

    # create bins and plot histograms
    bins = np.arange(lowerlim, upperlim + binwidth, binwidth)
    axHistx.hist(x[0, :], bins=bins, normed=1, color='navy')
    axHisty.hist(x[1, :], bins=bins, orientation='horizontal', normed=1, color='navy')

    # plot best-fit line, if target_PDF is given
    if target_PDF != 0:
        f_x0 = compute_marginal_PDF(target_PDF, bins, 0)
        axHistx.plot(bins, f_x0, '--', color='red')

        f_x1 = compute_marginal_PDF(target_PDF, bins, 1)
        axHisty.plot(f_x1, bins, '--', color='red')

    # limit histograms to limits of scatter-plot
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    # tight layout not possible here !
    plt.savefig('plot_scatter_with_hist.pdf', format='pdf', dpi=50, bbox_inches='tight')


# -----------------------------------------------------------------------------------------
# Helper function to compute the marginal pdf of a 2D multivariate pdf
def compute_marginal_PDF(target_PDF, bins, dimension):
    x_from  = -10
    x_till  = 10
    n_steps = 100.0

    dx      = (x_till - x_from) / n_steps
    x       = np.linspace(x_from, x_till, n_steps)

    len_x       = len(x)
    len_bins    = len(bins)
    y           = np.zeros((len_bins), float)

    # integrate over x1 to obtain marginal of x0
    if dimension == 0:
        for i in range(0, len_bins):
            for j in range(0, len_x):
                temp_x = bins[i]
                temp_y = x[j]
                y[i]  += target_PDF([temp_x, temp_y])

            y[i] = y[i]*dx

    # integrate over x2 to obtain marginal of x1
    if dimension == 1:
        for i in range(0, len_bins):
            for j in range(0, len_x):
                temp_y = bins[i]
                temp_x = x[j]
                y[i]  += target_PDF([temp_x, temp_y])

            y[i] = y[i]*dx

    return y


# ---------------------------------------------------------------------------
# plot cov over pf
def plot_cov_over_pf(pf_line, cov_mmh, cov_cs, cov_acs):
     # create figure
    fig = plt.figure()

    # set x-axis to log-scale
    plt.xscale('log')
    plt.gca().invert_xaxis()

    # plotting

    # * plot point of estimation of failure probability
    plt.plot(pf_line, cov_mmh, marker='s', color='navy',\
                    markersize='10', label=r'MMH')

    # * plot point of estimation of failure probability
    plt.plot(pf_line, cov_cs, marker='o', color='red',\
                    markersize='10', label=r'CS')
    
    # * plot point of estimation of failure probability
    plt.plot(pf_line, cov_acs, marker='x', color='green',\
                    markersize='10', label=r'aCS')

    # add legend
    matplotlib.rcParams['legend.fontsize'] = 12
    plt.legend(loc='upper right')

    # set labels
    plt.xlabel(r'Target Probability of Failure $P_f$')
    plt.ylabel(r'Coefficient of Variation $\delta$')
    plt.tight_layout()
    #plt.savefig('plot_sus_estimation.pdf', format='pdf', dpi=50, bbox_inches='tight')


# ---------------------------------------------------------------------------
def plot_m_with_poisson_dist(m_list, pf):
    # set up the distribution of m
    m_array = np.asarray(m_list)
    m_dist = np.bincount(m_array)
    N = len(m_array.reshape(-1))
    m_dist = m_dist/N    

    # parameter for poisson distribution
    lam     = -np.log(pf)
    poisson = lambda k: lam**k * np.exp(-lam) / np.math.factorial(k)

    x = np.linspace(0, len(m_dist)-1, len(m_dist))
    y = np.zeros(len(m_dist), float)
    for i in range(0, len(m_dist)):
        y[i] = poisson(x[i])
    
    # create figure
    fig = plt.figure()

    # plot results
    plt.plot(y)
    plt.plot(m_dist)


# ---------------------------------------------------------------------------
def plot_pf_over_b(b_line_list, pf_line_list, legend_list):
    plt.figure()

    # initilize colors
    colors = ['blue', 'fuchsia', 'green', 'red', 'navy', 'skyblue', 'orange', 'yellow']

    # plot all lines
    for i in range(0, len(b_line_list)):
        plt.plot(b_line_list[i], pf_line_list[i], '--', color=colors[np.mod(i, len(colors))], label=legend_list[i])

    # set y-axis to log-scale
    plt.yscale('log')

    # add legend
    matplotlib.rcParams['legend.fontsize'] = 12
    plt.legend(loc='lower right')

    # set labels
    plt.xlabel(r'Limit state function values $b$')
    plt.ylabel(r'$P(g(x) \leq b)$')
    plt.tight_layout()
    #plt.savefig('plot_pf_over_b.pdf', format='pdf', dpi=50, bbox_inches='tight')


# ---------------------------------------------------------------------------   
def plot_cov_over_b(b_line_list, cov_line_list, legend_list):
    plt.figure()

    # initilize colors
    colors = ['blue', 'fuchsia', 'green', 'red', 'navy', 'skyblue', 'orange', 'yellow', 'black', 'brown']

    # plot all lines
    for i in range(0, len(cov_line_list)):
        plt.plot(b_line_list[i], cov_line_list[i], '--', color=colors[np.mod(i, len(colors))], label=legend_list[i])

    # add legend
    matplotlib.rcParams['legend.fontsize'] = 12
    plt.legend(loc='upper right')

    # set labels
    plt.xlabel(r'Limit state function values $b$')
    plt.ylabel(r'$C.O.V.$')
    plt.tight_layout()
    #plt.savefig('plot_cov_over_b.pdf', format='pdf', dpi=50, bbox_inches='tight')


# ---------------------------------------------------------------------------
def plot_cov_ober_pf(b_line_list, cov_line_list, legend_list):

    return True