"""
# ---------------------------------------------------------------------------
# Plotting function to plot the failure probability estimation of SuS
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
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import plots.help_plot as hplt

from matplotlib.pyplot import *
from matplotlib import rcParams
from matplotlib import ticker


# General settings: create figure object with LaTeX font
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = 22
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

# ---------------------------------------------------------------------------
# plot subset-simulation
def plot_sus(g, p0, N, pf_sus, analytical_CDF=0):
    # create figure
    fig = plt.figure()

    # inizialization
    Nc          = int(N*p0)
    n_levels    = len(g)

    Pf_line     = np.zeros((n_levels, Nc), float)
    b_line      = np.zeros((n_levels, Nc), float)

    Pf          = np.zeros(n_levels, float)
    b           = np.zeros(n_levels, float)

    # level 0
    Pf[0]       = p0
    b[0]        = np.percentile(np.sort(g[0]), p0*100)

    Pf_line[0, :] = np.linspace(p0,1,Nc)
    b_line[0, :]  = np.percentile(np.sort(g[0]), Pf_line[0, :]*100)

    # loop over levels
    for i in range(1, n_levels):
        g_sorted     = np.sort(g[i])

        Pf[i]        = Pf[i-1]*p0
        b[i]         = np.percentile(g_sorted, p0*100)

        Pf_line[i,:] = Pf_line[i-1,:]*p0
        b_line[i,:]  = np.percentile(g_sorted, Pf_line[0, :]*100)
    
    #reshape and sort the matrices
    Pf_line = np.asarray(Pf_line).reshape(-1)
    Pf_line = np.sort(Pf_line)

    b_line  = np.asarray(b_line).reshape(-1)
    b_line  = np.sort(b_line)

    # exact line and exact point (with analytical_CDF) 
    if analytical_CDF!=0:
        b_exact_line    = np.linspace(0, 7, 140)
        pf_exact_line   = analytical_CDF(b_exact_line)

        pf_exact_point  = analytical_CDF(0)        

    # set y-axis to log-scale
    plt.yscale('log')

    # plotting

    # * plot line of estimator
    plt.plot(b_line, Pf_line, '--', color='navy', label=r'SuS')

    # * plot exact line
    if analytical_CDF != 0:
        plt.plot(b_exact_line, pf_exact_line, '--', color='red', label=r'Exact')

    # * plot intermediate steps (b)
    plt.plot(b, Pf, 'x', color='yellow', label=r'Intermediate levels')

    # * plot point of estimation of failure probability
    plt.plot(0, pf_sus, 'x', label=r'Pf SuS')

    # * plot exact point
    if analytical_CDF != 0: 
        plt.plot(0, pf_exact_point, 'x', label=r'Pf Exact')

    # add legend
    matplotlib.rcParams['legend.fontsize'] = 12
    plt.legend(loc='lower right')
    

    # set titles
    plt.title(r'Failure probability estimate')
    plt.xlabel(r'Limit state function $g$')
    plt.ylabel(r'Failure probability $P_f$')

    # Tweak spacing to prevent clipping of ylabel
    plt.tight_layout()
    #plt.savefig('plot_sus_estimation.pdf', format='pdf', dpi=50, bbox_inches='tight')
