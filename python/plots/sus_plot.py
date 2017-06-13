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
