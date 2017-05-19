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
# References:
# 1."Bayesian post-processor and other enhancements of Subset Simulation
#   for estimating failure probabilities in high dimensions"
#    Konstantin M. Zuev, James L. Beck, Siu-Kui Au, Lambros S. Katafygiotis
#    Computers and Structures 92-93 (2015) 283-296
# ---------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

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
    plt.plot(b, Pf, marker='o', markerfacecolor='none', markeredgecolor='black',\
                    markersize='8', linestyle='none', label=r'Intermediate levels')

    # * plot point of estimation of failure probability
    plt.plot(0, pf_sus, marker='x', color='navy',\
                    markersize='10', linestyle='none', label=r'Pf SuS')

    # * plot exact point
    if analytical_CDF != 0:
        plt.plot(0, pf_exact_point, marker='x', color='red',\
                    markersize='10', linestyle='none', label=r'Pf Exact')

    # add legend
    matplotlib.rcParams['legend.fontsize'] = 12
    plt.legend(loc='lower right')

    # set titles
    plt.title(r'Failure probability estimate')
    plt.xlabel(r'Limit state function values $b$')
    plt.ylabel(r'$P(g(x) \leq b)$')
    plt.tight_layout()
    #plt.savefig('plot_sus_estimation.pdf', format='pdf', dpi=50, bbox_inches='tight')

# ---------------------------------------------------------------------------
# plot subset-simulation
def plot_sus_list(g_list, p0, N, pf_sus_array, analytical_CDF=0):
    # create figure
    fig = plt.figure()

    # some constants
    Nc    = int(N*p0)
    n_sim = len(g_list)

    #b_line_list = []

    n_levels = np.zeros(n_sim, int)

    max_levels = 0
    for i in range(0, n_sim):
        n_levels[i] = len(g_list[i])
        if n_levels[i] > max_levels:
            max_levels = n_levels[i]

    # set up Pf_line
    Pf_line     = np.zeros((max_levels, Nc), float)
    Pf_line[0, :] = np.linspace(p0, 1, Nc)
    for i in range(1, max_levels):
        Pf_line[i, :] = Pf_line[i-1, :]*p0
    
    # initialize
    b_line_mean_matrix = np.zeros((max_levels, Nc), float)
    b_line_sigma_matrix = np.zeros((max_levels, Nc), float)

    for level in range(0, max_levels):
        b_line_list = []
        for sim in range(0, n_sim):
            if n_levels[sim] > level:
                b_line      = np.zeros((1,Nc), float)
                g = g_list[sim]

                g_sorted     = np.sort(g[level])
                b_line[0, :]  = np.percentile(g_sorted, Pf_line[0, :]*100)
                
                b_line  = np.sort(b_line)
                b_line_list.append(b_line)
        
        b_line_matrix_temp = np.asarray(b_line_list)


        b_line_mean_matrix[level, :] = np.mean(b_line_matrix_temp, axis=0)
        b_line_sigma_matrix[level, :] = np.std(b_line_matrix_temp, axis=0)

    # reshape and sort the matrices

    Pf_line = np.asarray(Pf_line).reshape(-1)
    Pf_line = np.sort(Pf_line)

    b_line_mean_array = np.asarray(b_line_mean_matrix).reshape(-1)
    b_line_mean_array = np.sort(b_line_mean_array)

    b_line_sigma_array = np.asarray(b_line_sigma_matrix).reshape(-1)
    b_line_sigma_array = np.sort(b_line_sigma_array)

    b_line_max = b_line_mean_array + 5*b_line_sigma_array
    b_line_min = b_line_mean_array - 5*b_line_sigma_array

    # exact line and exact point (with analytical_CDF) 
    if analytical_CDF!=0:
        b_exact_line    = np.linspace(0, 7, 140)
        pf_exact_line   = analytical_CDF(b_exact_line)

        pf_exact_point  = analytical_CDF(0)        

    # set y-axis to log-scale
    plt.yscale('log')

    # plotting

    # * plot line of estimator
    plt.plot(b_line_mean_array, Pf_line, '--', color='navy', label=r'SuS mu')
    plt.plot(b_line_max, Pf_line, ':', color='navy', label=r'mu plus sigma')
    plt.plot(b_line_min, Pf_line, ':', color='navy', label=r'mu minus sigma')

    # * plot exact line
    if analytical_CDF != 0:
        plt.plot(b_exact_line, pf_exact_line, '--', color='red', label=r'Exact')

    # * plot intermediate steps (b)
    #plt.plot(b, Pf, marker='o', markerfacecolor='none', markeredgecolor='black',\
    #                markersize='8', linestyle='none', label=r'Intermediate levels')

    # * plot point of estimation of failure probability
    plt.plot(0, np.mean(pf_sus_array), marker='x', color='navy',\
                    markersize='10', linestyle='none', label=r'Pf SuS')

    # * plot exact point
    if analytical_CDF != 0:
        plt.plot(0, pf_exact_point, marker='x', color='red',\
                    markersize='10', linestyle='none', label=r'Pf Exact')

    # add legend
    matplotlib.rcParams['legend.fontsize'] = 12
    plt.legend(loc='lower right')

    # set titles
    plt.title(r'Failure probability estimate')
    plt.xlabel(r'Limit state function values $b$')
    plt.ylabel(r'$P(g(x) \leq b)$')
    plt.tight_layout()
    #plt.savefig('plot_sus_estimation.pdf', format='pdf', dpi=50, bbox_inches='tight')
