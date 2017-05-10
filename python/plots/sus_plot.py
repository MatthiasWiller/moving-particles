"""
Author: Matthias Willer 2017
"""

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import plots.help_plot as hplt

from matplotlib.pyplot import *
from matplotlib import rcParams
from matplotlib import ticker


# -----------------------------------------------------------------------------------------
# plot subset-simulation
def plot_sus(g, p0, N, pf_sus=0, analytical_CDF=0):
    print('start plotting')
    # inizialization
    #g0          = g[0]
    Nc          = int(N*p0)
    n_levels    = len(g)
    Pf_line     = np.zeros((n_levels, Nc), float)
    #Pf_line     = np.linspace(p0, 1, Nc)
    b_line      = np.zeros((n_levels, Nc), float)
    
    # (1.): g-line from subset-simulation
    Pf           = np.zeros(n_levels, float)
    b            = np.zeros(n_levels, float)
    Pf[0]        = p0
    b[0]         = np.percentile(np.sort(g[0]), p0)
    Pf_line[0, :] = np.linspace(p0,1,Nc)
    b_line[0, :]  = np.percentile(np.sort(g[0]), Pf_line[0, :]*100)

    for i in range(1, n_levels):
        g_sorted    = np.sort(g[i])
        Pf[i]        = Pf[i-1]*p0
        b[i]         = np.percentile(g_sorted, p0)
        Pf_line[i,:] = Pf_line[i-1,:]*p0
        b_line[i,:]  = np.percentile(g_sorted, Pf_line[1, :]*100)

    #Pf_line = np.squeeze(np.asarray(Pf_line))
    Pf_line = np.asarray(Pf_line).reshape(-1)
    Pf_line = np.sort(Pf_line)
    b_line  = np.asarray(b_line).reshape(-1)
    b_line = np.sort(b_line)

    # (2.): exact line and exact point (with analytical_CDF) 
    if analytical_CDF!=0:
        b_exact_line = np.linspace(0, 7, 140)
        pf_exact_line = analytical_CDF(b_exact_line)

        pf_exact_point = analytical_CDF(0)

    # (3.):

    # actual plotting
    fig = plt.figure()

    # create figure object with LaTeX font
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.size'] = 22
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rc('text', usetex=True)

    # set y-axis to log-scale
    plt.yscale('log')

    # plot lines
    plt.plot(b_exact_line, pf_exact_line, '--', color='red')
    plt.plot(b_line, Pf_line, '--', color='navy')

    plt.plot(b, Pf, 'x', color='yellow')

    # plot points
    plt.plot(0, pf_sus, 'x')
    plt.plot(0, pf_exact_point, 'x')

    plt.title(r'Subset-Simulation')
    plt.xlabel(r'$g$')
    plt.ylabel(r'$P_f$')

    # Tweak spacing to prevent clipping of ylabel
    plt.tight_layout()
    #plt.savefig('plot_hist.pdf', format='pdf', dpi=50, bbox_inches='tight')
