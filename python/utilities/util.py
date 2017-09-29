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
# Version 2017-07
# ---------------------------------------------------------------------------
"""

import numpy as np

# ---------------------------------------------------------------------------
def get_pf_line_and_b_line_from_MCS(g_list):
    # sort g_list to obtain b_line_temp
    b_line_temp  = np.sort(g_list)
    pf_line_temp = np.arange(1, len(b_line_temp)+1)/float(len(b_line_temp))

    # interpolate pf-values to corresponding b-values
    n_values = 100
    upper_b  = b_line_temp[-1]

    b_line   = np.linspace(1e-3, upper_b, n_values)
    pf_line  = np.interp(b_line, b_line_temp, pf_line_temp)

    return b_line, pf_line


# ---------------------------------------------------------------------------
def get_pf_line_and_b_line_from_SUS(g_list, p0, N):
    # some constants
    Nc    = int(N*p0)    # number of chains
    n_sim = len(g_list)  # number of simulations

    # find max n_levels
    n_levels = max(len(g_tmp) for g_tmp in g_list)

    # set up pf_line
    pf_line_temp       = np.zeros((n_levels, Nc), float)
    pf_line_temp[0, :] = np.linspace(p0, 1, Nc)
    for i in range(1, n_levels):
        pf_line_temp[i, :] = pf_line_temp[i-1, :] * p0
 
    # initialization
    b_line_list = []

    # loop over all simulations to get the b_line
    for sim in range(0, n_sim):
        g          = g_list[sim]

        n_levels   = len(g)

        b_line_temp = np.zeros((n_levels, Nc), float)

        # loop over all levels and get b_line
        for lvl in range(0, n_levels):
            g_sorted            = np.sort(g[lvl])
            b_line_temp[lvl, :] = np.percentile(g_sorted, pf_line_temp[0, :]*100)

        b_line_array_temp = b_line_temp.reshape(-1)
        b_line_array_temp = np.sort(b_line_array_temp)
        b_line_list.append(b_line_array_temp)

    # reshape and sort the matrices
    pf_line_temp = np.asarray(pf_line_temp).reshape(-1)
    pf_line_temp = np.sort(pf_line_temp)

    pf_line_list = []

    # interpolate pf_line-values to get a pf_line_list,
    # that is dependent only on one b_line
    n_values = 100
    upper_b  = min(max(np.amax(g_lvl) for g_lvl in g_sim) for g_sim in g_list)
    b_line   = np.linspace(1e-3, upper_b, n_values)

    for i in range(0, n_sim):
        len_b    = len(b_line_list[i])
        pf_line  = np.interp(b_line, b_line_list[i], pf_line_temp[-len_b:])
        pf_line_list.append(pf_line)

    return b_line, pf_line_list


# ---------------------------------------------------------------------------
def get_pf_line_and_b_line_from_MP(g_list_list, N):
    # get number of simulations
    n_sim = len(g_list_list)

    # initialization
    b_line_list_prelim  = []
    pf_line_list_prelim = []

    for i in range(0, n_sim):
        # sort g_list
        g_list = g_list_list[i]
        g_list.sort(reverse=True)

        # initialization
        pf_line_temp = np.zeros(len(g_list)-N)
        b_line_temp  = np.zeros(len(g_list)-N)

        for m in range(0, len(g_list)-N):
            pf_line_temp[m] = (1 -1/N)**(m+1)
            b_line_temp[m]  = g_list[m]
        
        pf_line_temp.sort()
        b_line_temp.sort()

        pf_line_list_prelim.append(pf_line_temp)
        b_line_list_prelim.append(b_line_temp)
    
    # initialization
    pf_line_list = []

    # interpolate pf_line-values to get a pf_line_list,
    # that is dependent only on one b_line
    n_values = 100
    upper_b  = min(max(g_sim) for g_sim in g_list_list)
    b_line   = np.linspace(1e-3, upper_b, n_values)

    for i in range(0, n_sim):
        pf_line  = np.interp(b_line, b_line_list_prelim[i], pf_line_list_prelim[i])
        pf_line_list.append(pf_line)

    return b_line, pf_line_list


# ---------------------------------------------------------------------------
def get_mean_and_cov_from_pf_lines(pf_line_list):
    # get number of simulations
    n_sim = len(pf_line_list)

    # turn lists in a matrix to get mean and std
    pf_line_matrix = np.asarray(pf_line_list).reshape(n_sim, -1)

    # get mean and std 
    pf_line_mean = np.mean(pf_line_matrix, axis=0)
    pf_line_std  = np.std(pf_line_matrix, axis=0)
    pf_line_cov  = pf_line_std / pf_line_mean

    return pf_line_mean, pf_line_cov

# ---------------------------------------------------------------------------
def get_ncall_lines_and_pf_line_from_MP(b_line_analytical, pf_line_analytical, g_list_list, N, Nb):
    n_sim = len(g_list_list)

    # initialization
    ncall_line_list = []
    pf_line = pf_line_analytical

    for i in range(0, n_sim):
        g_list = g_list_list[i]
        b_line = np.flipud(np.sort(g_list))
        # b_line = b_line[::-1] # reverse array
        m = len(g_list) - N
        ncall_line = np.zeros(m)

        for j in range(0,m):
            ncall_line[j] = N + (j+1)*Nb
        b_line = b_line[N:]

        ncall_line_new = np.interp(b_line_analytical, np.flipud(b_line), np.flipud(ncall_line) )
        ncall_line_list.append(ncall_line_new)

    return pf_line, ncall_line_list

# ---------------------------------------------------------------------------
def get_ncall_lines_and_cov_line_from_MP(pf_line_list):
    print('function is empty!')

# ---------------------------------------------------------------------------
def get_mean_ncall_from_MP(g_list_list, number_of_samples, Nb):
    nsim = len(g_list_list)
    ncall_array = np.zeros(nsim)
    for i in range(0, nsim):
        nele = len(g_list_list[i])
        ncall_array[i] = (nele - number_of_samples)*Nb + number_of_samples

    return np.mean(ncall_array)


# ---------------------------------------------------------------------------
def get_mean_ncall_from_SUS(g_list_list, number_of_samples_per_level, p0):
    nsim = len(g_list_list)
    ncall_array = np.zeros(nsim)
    for i in range(0, nsim):
        nlvl = len(g_list_list[i])
        ncall_array[i] = (nlvl - 1)*(1-p0)*number_of_samples_per_level + number_of_samples_per_level

    return np.mean(ncall_array)


# ---------------------------------------------------------------------------
def get_mean_and_cov_pf_from_MP(g_list_list, number_of_samples):
    nsim = len(g_list_list)
    pf_array = np.zeros(nsim)
    for i in range(0, nsim):
        m = len(g_list_list[i]) - number_of_samples
        pf_array[i] = (1 - 1/number_of_samples)**m

    pf_mean = np.mean(pf_array)
    pf_cov = np.std(pf_array)/pf_mean
    return pf_mean, pf_cov


# ---------------------------------------------------------------------------
def get_mean_and_cov_pf_from_SUS(g_list_list, number_of_samples_per_level, p0):
    nsim = len(g_list_list)
    pf_array = np.zeros(nsim)
    for i in range(0, nsim):
        nlvl = len(g_list_list[i])
        g_lvl = np.array(g_list_list[i][-1])
        n_fail = np.ones(number_of_samples_per_level, float)
        n_fail = n_fail[g_lvl<0]
        pf_array[i] = p0**(nlvl-1)*np.sum(n_fail)/number_of_samples_per_level

    pf_mean = np.mean(pf_array)
    pf_cov = np.std(pf_array)/pf_mean
    return pf_mean, pf_cov