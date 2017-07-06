"""
# ---------------------------------------------------------------------------
# Several utilities functions
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

# ---------------------------------------------------------------------------
def get_n_eff_sim(g_list):
    # initialization
    n_sim    = len(g_list)
    n_levels = np.zeros(n_sim, int)

    # count number of levels for every simulation
    for i in range(0, n_sim):
        n_levels[i] = len(g_list[i])

    # find most often encountered n_levels
    count_n_levels   = np.bincount(n_levels)
    most_often_level = np.nanargmax(count_n_levels)
    #n_levels         = most_often_level
    n_eff_sim = count_n_levels[most_often_level]

    return n_eff_sim

# ---------------------------------------------------------------------------
def get_pf_line_and_b_line_from_SUS(g_list, p0, N):
    # some constants
    Nc    = int(N*p0)    # number of chains
    n_sim = len(g_list)  # number of simulations

    # initialization
    n_levels = np.zeros(n_sim, int)

    # count number of levels
    for i in range(0, n_sim):
        n_levels[i] = len(g_list[i])

    # find max n_levels
    n_levels = np.amax(n_levels)

    # set up Pf_line
    pf_line       = np.zeros((n_levels, Nc), float)
    pf_line[0, :] = np.linspace(p0, 1, Nc)
    for i in range(1, n_levels):
        pf_line[i, :] = pf_line[i-1, :] * p0
    
    # initialization
    b_line_list = []

    # loop over all simulations to get the b_line
    for sim in range(0, n_sim):
        g           = g_list[sim]

        n_levels    = len(g)

        b_line      = np.zeros((n_levels, Nc), float)

        # loop over all levels and get b_line
        for level in range(0, n_levels):
            g_sorted          = np.sort(g[level])
            b_line[level, :]  = np.percentile(g_sorted, pf_line[0, :]*100)

        b_line_array_temp = b_line.reshape(-1)
        b_line_array_temp = np.sort(b_line_array_temp)
        b_line_list.append(b_line_array_temp)

    # reshape and sort the matrices
    pf_line = np.asarray(pf_line).reshape(-1)
    pf_line = np.sort(pf_line)

    return b_line_list, pf_line

# ---------------------------------------------------------------------------
def get_pf_line_and_b_line_from_MP(g_list_list, N):
    # get number of simulations
    n_sim = len(g_list_list)

    # initialization
    b_line_list = []
    pf_line_list = []

    for i in range(0, n_sim):
        # sort g_list
        g_list = g_list_list[i]
        g_list.sort(reverse=True)

        # initialization
        pf_line = np.zeros(len(g_list)-N)
        b_line = np.zeros(len(g_list)-N)

        for m in range(0, len(g_list)-N):
            pf_line[m] = (1 -1/N)**(m+1)
            b_line[m] = g_list[m]
        
        pf_line_list.append(pf_line)
        b_line_list.append(b_line)

    return b_line_list, pf_line_list

# ---------------------------------------------------------------------------
def get_pf_line_and_b_line_from_MCS(g_list):
    # sort g_list to obtain b_line
    b_line_mcs  = np.sort(g_list)
    pf_line_mcs = np.arange(1, len(b_line_mcs)+1)/float(len(b_line_mcs))

    # delete all negative values
    n_negative_values = sum(1 for g in g_list if g < 0)

    b_line_mcs  = b_line_mcs[n_negative_values:]
    pf_line_mcs = pf_line_mcs[n_negative_values:]

    return b_line_mcs, pf_line_mcs
