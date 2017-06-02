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
# Version 2017-05
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
        y = target_PDF(bins)
        plt.plot(bins, y, '--', color='red')

    if((target_PDF != 0) and (dimension == 2)):
        # for 2D case
        y = compute_marginal_PDF(target_PDF, bins, 0)
        plt.plot(bins, y, '--', color='red')

    plt.ylim(0,1)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.title(r'Histogram of $\theta$')
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'Frequency $p$')
    plt.tight_layout()
    #plt.savefig('plot_hist.pdf', format='pdf', dpi=50, bbox_inches='tight')


# -----------------------------------------------------------------------------------------
# plot values of index
def plot_mixing(x):
    
    n_samples = len(x)

    plt.figure()
    plt.plot(x, color='navy')

    plt.title(r'Iterations of the MCMC')
    plt.xlabel(r'Number of samples, $n$')
    plt.ylabel(r'$\theta_1$')
    plt.xlim(0, n_samples)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.tight_layout()
    #plt.savefig('plot_mixing.pdf', format='pdf', dpi=50, bbox_inches='tight')

# -----------------------------------------------------------------------------------------
# plot of the estimated autocorrelation of samples
def plot_autocorr(x, lag):

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
    #rho = hplt.estimate_autocorrelation(x, lag)    

    # plot results
    plt.figure()
    plt.plot(rho, '.')

    plt.title(r'Sample autocorrelation')
    plt.xlabel(r'Lag, $k$')
    plt.ylabel(r'$\hat{R}(k)')
    plt.xlim(0, lag)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.tight_layout()
    #plt.savefig('plot_autocorr.pdf', format='pdf', dpi=50, bbox_inches='tight')

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
    plt.title(r'Contour plot')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.xticks([-2, 0, 2, 4, 6, 8])
    plt.yticks([-2, 0, 2, 4, 6, 8])

    plt.gca().set_aspect('equal', adjustable='box')
    #plt.tight_layout()
    #plt.savefig('plot_scatter_with_contour.pdf', format='pdf', dpi=50, bbox_inches='tight')

# -----------------------------------------------------------------------------------------
# plot surface with samples
def plot_surface_with_samples(theta, f):
    x       = np.linspace(-2, 6, 100)
    X, Y    = np.meshgrid(x, x)
    Z       = f([X, Y])


    z_plane = 7.5 * np.ones(len(x))

    z_samples = f([theta[:, 0], theta[:, 1]])


    theCM = cm.get_cmap()
    theCM._init()
    alphas = np.abs(np.linspace(-1.0, 1.0, theCM.N))
    theCM._lut[:-3,-1] = alphas

    # 3D Plot
    min_x = min(X.flatten())
    min_y = min(Y.flatten())
    min_z = min(Z.flatten())

    max_x = max(X.flatten())
    max_y = max(Y.flatten())
    max_z = max(Z.flatten())

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_surface(X, Y, Z, rstride=5, cstride=5, cmap=cm.pink_r, antialiased=False, alpha=0.9)
    ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5, linewidth=0.5, color='black', alpha=0.9)

    ax.scatter(theta[:, 0], theta[:, 1], z_samples, marker='o', color='blue', label='$z(x_1, x_2)$')

    ax.plot_surface(X, Y, z_plane, rstride=5, cstride=5, cmap=theCM, antialiased=False, alpha=0.1)

    ax.view_init(elev=24, azim=-40)

    # axes and title config
    ax.set_xlabel('$x_1$', labelpad=15)
    ax.yaxis.set_rotate_label(False) # disable automatic rotation
    ax.set_ylabel('$x_2$', rotation = 15, labelpad=15)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('$z(x)$',rotation=93, labelpad=3)
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

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.pink)

    ttl = ax.set_title('Target distribution')
    ttl.set_position([.5, 0.95])
    ax.view_init(elev=45, azim=40)
    ax.set_xlim3d(-2, 8)
    ax.set_ylim3d(-2, 8)
    ax.set_zlim3d(0, 0.25)
    ax.xaxis._axinfo['label']['space_factor'] = 20
    ax.yaxis._axinfo['label']['space_factor'] = 20
    ax.zaxis._axinfo['label']['space_factor'] = 20

    plt.tight_layout()
    #plt.savefig('plot3d.pdf', format='pdf', dpi=50, bbox_inches='tight')


# -----------------------------------------------------------------------------------------
# plot combination of Scatter plot with histogram
def plot_scatter_with_hist(x, target_PDF=0):

    nullfmt = NullFormatter()         # no labels

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
    axHistx.hist(x[0,:], bins=bins, normed=1, color='navy')
    axHisty.hist(x[1,:], bins=bins, orientation='horizontal', normed=1, color='navy')
    
    # plot best-fit line, if target_PDF is given
    if(target_PDF != 0):
        f_x0 = compute_marginal_PDF(target_PDF, bins, 0)
        axHistx.plot(bins, f_x0, '--', color='red')

        f_x1 = compute_marginal_PDF(target_PDF, bins, 1)
        axHisty.plot(f_x1, bins, '--', color='red')
    
    # limit histograms to limits of scatter-plot
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    # tight layout not possible here !
    #plt.savefig('plot_scatter_with_hist.pdf', format='pdf', dpi=50, bbox_inches='tight')

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
# plot subset-simulation
def plot_sus_list(g_list, p0, N, pf_sus_array, analytical_CDF=0):
    # create figure
    fig = plt.figure()

    # some constants
    Nc    = int(N*p0)
    n_sim = len(g_list)

    # initialization
    n_levels = np.zeros(n_sim, int)

    # count number of levels
    for i in range(0, n_sim):
        n_levels[i] = len(g_list[i])

    # find most often encountered n_levels
    count_n_levels   = np.bincount(n_levels)
    most_often_level = np.nanargmax(count_n_levels)
    n_levels         = most_often_level

    # delete all other levels
    for i in reversed(range(0, n_sim)):
        if len(g_list[i]) != most_often_level:
            g_list.pop(i)
    
    n_sim_effective = len(g_list)

    print('The number of effective samples was successfully reduced from', n_sim, 'to', n_sim_effective, '!')

    # set up Pf_line
    Pf_line       = np.zeros((n_levels, Nc), float)
    Pf_line[0, :] = np.linspace(p0, 1, Nc)
    for i in range(1, n_levels):
        Pf_line[i, :] = Pf_line[i-1, :]*p0
    
    # initialize matrices and list
    b_line_mean_matrix  = np.zeros((n_levels, Nc), float)
    b_line_sigma_matrix = np.zeros((n_levels, Nc), float)

    b_line_list_all_levels = []

    # loop over all (effective) simulations to get the b_line
    for sim in range(0, n_sim_effective):
        b_line_list = []
        g = g_list[sim]

        b_line      = np.zeros((n_levels, Nc), float)

        # loop over all levels and get b_line
        for level in range(0, n_levels):
            g_sorted          = np.sort(g[level])
            b_line[level, :]  = np.percentile(g_sorted, Pf_line[0, :]*100)
        
        b_line_array_temp = b_line.reshape(-1)
        b_line_array_temp = np.sort(b_line_array_temp)
        b_line_list_all_levels.append(b_line_array_temp)
    
    # reshape and sort the matrices
    Pf_line = np.asarray(Pf_line).reshape(-1)
    Pf_line = np.sort(Pf_line)

    b_line_matrix = np.asarray(b_line_list_all_levels)

    b_line_mean_array = np.mean(b_line_matrix, axis=0)
    b_line_sigma_array = np.std(b_line_matrix, axis=0)

    b_line_max = b_line_mean_array + 5*b_line_sigma_array
    b_line_min = b_line_mean_array - 5*b_line_sigma_array

    # exact line and exact point (with analytical_CDF) 
    if analytical_CDF!=0:
        max_lim         = np.max(np.asarray(g))
        b_exact_line    = np.linspace(0, max_lim, 140)
        pf_exact_line   = analytical_CDF(b_exact_line)

        pf_exact_point  = analytical_CDF(0)        

    # set y-axis to log-scale
    plt.yscale('log')

    # plotting

    # * plot exact line
    if analytical_CDF != 0:
        plt.plot(b_exact_line, pf_exact_line, '-', color='red', label=r'Exact')

    # * plot line of estimator
    plt.plot(b_line_mean_array, Pf_line, '--', color='navy', label=r'SuS mu')
    label_text = r'$\mu \pm 5\sigma$ (' + repr(n_sim_effective) + r' sim)'
    plt.fill_betweenx(Pf_line, b_line_min, b_line_max, color='powderblue', label=label_text)    

    # * plot intermediate steps (b)
    #plt.plot(b, Pf, marker='o', markerfacecolor='none', markeredgecolor='black',\
    #                markersize='8', linestyle='none', label=r'Intermediate levels')

    # * plot exact point
    if analytical_CDF != 0:
        plt.plot(0, pf_exact_point, marker='x', color='red',\
                    markersize='10', linestyle='none', label=r'Pf Exact')

    # * plot point of estimation of failure probability
    plt.plot(0, np.mean(pf_sus_array), marker='x', color='navy',\
                    markersize='10', linestyle='none', label=r'Pf SuS')

    # add legend
    matplotlib.rcParams['legend.fontsize'] = 12
    plt.legend(loc='lower right')

    # set titles
    plt.title(r'Failure probability estimate')
    plt.xlabel(r'Limit state function values $b$')
    plt.ylabel(r'$P(g(x) \leq b)$')
    plt.tight_layout()
    #plt.savefig('plot_sus_estimation.pdf', format='pdf', dpi=50, bbox_inches='tight')
    