"""
Author: Matthias Willer 2017
"""

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

def get_marginal_PDF(target_PDF, bins):
    x_from = -10
    x_till = 10
    n_steps = 50.0
    dx = (x_till - x_from) / n_steps
    x = np.linspace(x_from, x_till, n_steps)

    len_x = len(x)
    len_bins = len(bins)

    y = np.zeros((len_bins), float)
    for i in range(0, len_bins):
        for j in range(0, len_x):
            temp_x = bins[i]
            temp_y = x[j]
            y[i] += target_PDF([temp_x, temp_y])
        
        y[i] = y[i]*dx

    return y

# histogram plot
def hist_plot(x, target_PDF=0):
    num_bins = 100 # default = 50

    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(x[0,:], num_bins, normed=1)

    # dimension of the problem
    d = x.shape[0]

    # add a 'best fit' line
    if((target_PDF != 0) and (d != 2)):
        y = target_PDF(bins)
        ax.plot(bins, y, '--')
    
    if((target_PDF != 0) and (d == 2)):
        y = get_marginal_PDF(target_PDF, bins)
        ax.plot(bins, y, '--')

    ax.set_xlabel('theta')
    ax.set_ylabel('p')
    ax.set_title(r'Histogram of theta')

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()


# plot values of index
def n_plot(x):
    fig, ax = plt.subplots()
    if np.size(x) > 1:
        ax.plot(x[0,:])
    else:
        ax.plot(x)
    
    ax.set_xlabel('n')
    ax.set_ylabel('theta')
    ax.set_title(r'Plot of theta')

    fig.tight_layout()

# plot of the estimated autocorrelation of samples
def estimated_autocorrelation(x, n_samples):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """

    # number of samples to use
    n = np.minimum(len(x),n_samples)
    x = x[:n]

    # compute variance and mean-value
    variance = x.var()
    x = x-x.mean()

    # compute autocorrelation (for explanation see stackoverflow/wikipedia)
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))

    fig, ax = plt.subplots()

    # plot results
    ax.stem(result)
    ax.set_title(r'Autocorrelation')
    ax.set_xlabel('n')
    ax.set_ylabel('autocorrelation')

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()


# make a nice contour plot    
def contour_plot(theta, target_PDF=0):
    fig, ax = plt.subplots()
    
    #fig =  plt.figure()
    #ax = fig.gca(projection='3d')

    # plot results
    ax.scatter(theta[0,:], theta[1,:])
    #ax.plot_surface(theta[0,:], theta[1,:], target_PDF(theta) )
    ax.set_title(r'Contour plot')
    #ax.set_xlabel('n')
    #ax.set_ylabel('autocorrelation')

    X = np.arange(-2, 8, 0.25)
    xlen = len(X)
    Y = np.arange(-2, 9, 0.25)
    ylen = len(Y)
    X, Y = np.meshgrid(X, Y)
    Z = target_PDF([X,Y])
    ax.contour(X,Y,Z)


    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()

def surface_plot(target_PDF):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = np.arange(-2, 8, 0.25)
    xlen = len(X)
    Y = np.arange(-2, 9, 0.25)
    ylen = len(Y)
    X, Y = np.meshgrid(X, Y)
    Z = target_PDF([X,Y])
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False)
    