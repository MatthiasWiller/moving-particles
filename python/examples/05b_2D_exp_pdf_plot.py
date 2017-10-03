"""
# ---------------------------------------------------------------------------
# Metropolis-Hastings algorithm 2D example: Example 6.4 Ref. [1]
# ---------------------------------------------------------------------------
# Created by:
# Matthias Willer (matthias.willer@tum.de)
# Engineering Risk Analysis Group
# Technische Universitat Munchen
# www.era.bgu.tum.de
# ---------------------------------------------------------------------------
# Version 2017-05
# ---------------------------------------------------------------------------
# References:
# 1."Simulation and the Monte Carlo method"
#    Rubinstein and Kroese (2017)
# ---------------------------------------------------------------------------
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import utilities.plots as uplt
import utilities.stats as ustat

import matplotlib
import matplotlib.pyplot as plt

from matplotlib.pyplot import cm
from matplotlib import rcParams
from matplotlib import ticker
from matplotlib.ticker import NullFormatter

from mpl_toolkits.mplot3d import Axes3D

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = 22
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

savepdf = True


# load data
# proposal = 'cs'
proposal = 'mh'
theta = np.load('python/data/samples_2D_exp_pdf_'+ proposal +'.npy')


# define target pdf

# target pdf 
c = 0 # correlation rho = 0
# c = 1 # correlation rho = -0.40
target_PDF = lambda x: (1 - c*(1-x[0]-x[1])+c*c*x[0]*x[1])*np.exp(-(x[0]+x[1]+c*x[0]*x[1]))


# OUTPUT
print('E[X1] =', round(theta[0,:].mean(), 5))
print('E[X2] =', round(theta[1,:].mean(), 5))

ustat.get_acceptance_rate(theta)

# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

# MIXING --------------------------------------------------------------------
matplotlib.rcParams['font.size'] = 24
n_samples = 1000

plt.figure()
plt.plot(theta[0,:1000], color='navy')

# set labels
plt.xlabel(r'Number of samples, $n$')
plt.ylabel(r'$x_1$')
plt.xlim(0, n_samples)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

plt.tight_layout()
if savepdf:
    plt.savefig('plot_mixing_' + proposal + '.pdf', format='pdf', dpi=50, bbox_inches='tight')

# AUTOCORRELATION 1 --------------------------------------------------------------------
matplotlib.rcParams['font.size'] = 24

# compute sample autocorrelation
lag = 50
x = theta[0,:]
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
plt.plot(rho[:20], '.')

# set labels
plt.xlabel(r'Lag, $k$')
plt.ylabel(r'$\hat{R}(k)')
plt.xlim(0, lag)
plt.ylim(0, 1)
plt.yticks([0, 0.5, 1])
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.tight_layout()
if savepdf:
    plt.savefig('plot_autocorr_1_' + proposal + '.pdf', format='pdf', dpi=50, bbox_inches='tight')


# AUTOCORRELATION 2 --------------------------------------------------------------------
matplotlib.rcParams['font.size'] = 24

# compute sample autocorrelation
lag = 50
x = theta[1,:]
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
plt.plot(rho[:20], '.')

# set labels
plt.xlabel(r'Lag, $k$')
plt.ylabel(r'$\hat{R}(k)')
plt.xlim(0, lag)
plt.ylim(0, 1)
plt.yticks([0, 0.5, 1])
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.tight_layout()
if savepdf:
    plt.savefig('plot_autocorr_2_'+proposal+'.pdf', format='pdf', dpi=50, bbox_inches='tight')


# PLOT SCATTER WITH HISTOGRAMS ----------------------------------------------------------------
x = theta[:,:5000]
nullfmt = NullFormatter()         # no labels

n_grid = 100
xx       = np.linspace(0, 5, n_grid)
X, Y    = np.meshgrid(xx, xx)
Z = np.zeros((n_grid, n_grid))
for i in range(0, len(X)):
    for j in range(0, len(Y)):
        Z[i,j] = target_PDF(np.array([X[i,j], Y[i,j]]))


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
upperlim = 8

# set limits of scatter plot
axScatter.set_xlim((lowerlim, upperlim))
axScatter.set_ylim((lowerlim, upperlim))

# create bins and plot histograms
bins = np.arange(lowerlim, upperlim + binwidth, binwidth)
axHistx.hist(x[0, :], bins=bins, normed=1, color='navy')
axHisty.hist(x[1, :], bins=bins, orientation='horizontal', normed=1, color='navy')

# plot best-fit line, if target_PDF is given
if target_PDF != 0:
    f_x0 = uplt.compute_marginal_PDF(target_PDF, bins, 0)
    axHistx.plot(bins, f_x0, '--', color='red')

    f_x1 = uplt.compute_marginal_PDF(target_PDF, bins, 1)
    axHisty.plot(f_x1, bins, '--', color='red')

# limit histograms to limits of scatter-plot
axHistx.set_xlim(axScatter.get_xlim())
axHisty.set_ylim(axScatter.get_ylim())

# set labels
axHistx.set_ylabel(r'$f_{X_2}(x_2)$')
axHisty.set_xlabel(r'$f_{X_1}(x_1)$')

axScatter.set_xlabel(r'$x_1$')
axScatter.set_ylabel(r'$x_2$')

# set ticks and limits
axHistx.set_yticks([0, 0.5])
axHistx.set_ylim([0, 0.8])

axHisty.set_xticks([0,0.5])
axHisty.set_xlim([0,0.8])

axScatter.set_xticks([0, 5])
axScatter.set_yticks([0, 5])

if savepdf:
    # tight layout not possible here !
    plt.savefig('plot_scatter_with_hist_' + proposal + '.pdf', format='pdf', dpi=50, bbox_inches='tight')



# PLOT PDF in 3D ---------------------------------------------------------------------
fig = plt.figure()

ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=20, cstride=20, cmap=cm.pink_r, antialiased=False, alpha=1.0)
ax.plot_wireframe(X, Y, Z, rstride=20, cstride=20, linewidth=0.5, color='black', alpha=1.0)
ax.view_init(elev=35, azim=-35)

# axes and title config
axx = ax.set_xlabel('$x_1$', labelpad=15)
ax.xaxis.set_rotate_label(False) # disable automatic rotation
axy = ax.set_ylabel('$x_2$', rotation = 0, labelpad=15)
ax.yaxis.set_rotate_label(False)
axz = ax.set_zlabel('$f(x_1, x_2)$',rotation=93, labelpad=7)
ax.zaxis.set_rotate_label(False)
# ax.set_xlim3d(min_x, max_x)
# ax.set_ylim3d(min_y, max_y)
ax.set_xticks([0, 5])
ax.set_yticks([0, 5])
ax.set_zticks([0.5, 1.0])

if savepdf:
    plt.savefig('plot_multivariate_PDF_3D.pdf', format='pdf', dpi=50, bbox_inches='tight', pad_inches=0.4)


plt.show()