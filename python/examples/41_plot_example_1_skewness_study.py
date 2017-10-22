"""
# ---------------------------------------------------------------------------
# File to produce plots for example 1
# ---------------------------------------------------------------------------
# Created by:
# Matthias Willer (matthias.willer@tum.de)
# Engineering Risk Analysis Group
# Technische Universitat Munchen
# www.era.bgu.tum.de
# ---------------------------------------------------------------------------
# Version 2017-07
# ---------------------------------------------------------------------------
# References:
# 1."MCMC algorithms for Subset Simulation"
#    Papaioannou, Betz, Zwirglmaier, Straub (2015)
# ---------------------------------------------------------------------------
"""

import time as timer

import numpy as np
import scipy.stats as scps
import matplotlib.pyplot as plt

import utilities.plots as uplt
import utilities.util as uutil

print("RUN 41_plot_example_1.py")

# set seed for randomization
np.random.seed(0)

savepdf = True

# ---------------------------------------------------------------------------
# STANDARD INPUT
# ---------------------------------------------------------------------------

# parameters
d    = 10        # number of dimensions

Nsus = 1000      # SUS: number of samples per conditional level
p0   = 0.1       # SUS: Probability of each subset, chosen adaptively

Nmp  = 100       # MP: Number of initial samples 

pf_list = ['1e-3', '1e-6', '1e-9', '1e-12']

def ecdf(data):
    n = len(data) # Number of data points: n
    x = np.sort(data) # x-data for the ECDF: x
    y = np.arange(1, n+1) / n # y-data for the ECDF: y

    return x, y

# ---------------------------------------------------------------------------
# LOAD RESULTS FROM SIMULATIONS
# ---------------------------------------------------------------------------
direction = 'python/data/'

for pf in pf_list:
    g_list_sus = np.load(direction + 'sus_example_1_d10_N1000_Nsim100_cs_pf' + pf + '_g_list.npy')
    g_list_mp = np.load(direction + 'mp_example_1_d10_N100_Nsim100_b5_cs_sss2_pf' + pf + '_g_list.npy')


    # ---------------------------------------------------------------------------
    # POST-PROCESSING
    # ---------------------------------------------------------------------------

    pf_array_sus = uutil.get_pf_array_from_SUS(g_list_sus, Nsus, p0)
    pf_array_mp  = uutil.get_pf_array_from_MP(g_list_mp, Nmp)


    # ---------------------------------------------------------------------------
    # PLOTS
    # ---------------------------------------------------------------------------
    fig = plt.figure()
    x_mp, y_mp = ecdf(pf_array_mp)
    x_sus, y_sus = ecdf(pf_array_sus)

    plt.plot(x_mp,y_mp, color='C1', label='MP')
    plt.plot(x_sus,y_sus, color='C2', label='SuS')
    plt.xscale('log')
    plt.xlabel(r'$p_f$')
    plt.ylabel(r'$P(P_f<p_f)$')

    plt.legend()

    plt.tight_layout()
    if savepdf:
        plt.savefig('skewness_study_example_1_pf'+pf+'.pdf', format='pdf', dpi=50, bbox_inches='tight')


plt.show()
