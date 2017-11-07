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

import numpy as np
import scipy.stats as scps
import matplotlib
import matplotlib.pyplot as plt

import utilities.util as uutil

# create figure object with LaTeX font
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = 22
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

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
# pf_list = ['1e-12']
minmax_list = [[-4,-2],[-7, -5],[-11, -7],[-14, -10]]

def ecdf(data):
    n = len(data) # Number of data points: n
    x = np.sort(data) # x-data for the ECDF: x
    y = np.arange(1, n+1) / n # y-data for the ECDF: y

    return x, y

# ---------------------------------------------------------------------------
# LOAD RESULTS FROM SIMULATIONS
# ---------------------------------------------------------------------------
direction = 'python/data/'
ii = 0
for pf in pf_list:
    g_list_sus = np.load(direction + 'sus_example_1_d10_N1000_Nsim100_cs_pf' + pf + '_g_list.npy')
    g_list_mp = np.load(direction + 'mp_example_1_d10_N100_Nsim100_b5_cs_sss2_pf' + pf + '_g_list.npy')


    # ---------------------------------------------------------------------------
    # POST-PROCESSING
    # ---------------------------------------------------------------------------

    pf_array_sus = uutil.get_pf_array_from_SUS(g_list_sus, Nsus, p0)
    pf_array_mp  = uutil.get_pf_array_from_MP(g_list_mp, Nmp)

    x_line_pdf = np.linspace(min(pf_array_sus), max(pf_array_sus), 100)
    kde_sus = scps.gaussian_kde(pf_array_sus)
    kde_mp  = scps.gaussian_kde(pf_array_mp)


    # ---------------------------------------------------------------------------
    # PLOTS
    # ---------------------------------------------------------------------------
    fig = plt.figure()
    x_mp, y_mp = ecdf(pf_array_mp)
    x_sus, y_sus = ecdf(pf_array_sus)

    plt.plot(x_sus,y_sus, color='C2', label='SuS')
    plt.plot(x_mp,y_mp, color='C1', label='MP')
    plt.xscale('log')
    plt.xlabel(r'$p_f$')
    plt.ylabel(r'$P(P_f<p_f)$')

    plt.legend()

    plt.tight_layout()
    if savepdf:
        plt.savefig('skewness_study_example_1_pf'+pf+'_cdf.pdf', format='pdf', dpi=50, bbox_inches='tight')


    fig = plt.figure()
    y_pdf_sus = kde_sus(x_line_pdf)
    y_pdf_mp = kde_mp(x_line_pdf)
    data = np.vstack([y_pdf_sus, y_pdf_mp]).T

    plt.hist([pf_array_sus, pf_array_mp], \
            bins=np.logspace(minmax_list[ii][0], \
            minmax_list[ii][1], 20), \
            color=['C2', 'C1'], \
            label=['SuS', 'MP'])
    plt.xscale('log')

    # plt.plot(x_line_pdf, y_pdf_mp, color='C1', label='MP')
    # plt.plot(x_line_pdf, y_pdf_sus, color='C2', label='SuS')
 

    plt.xlabel(r'$p_f$')
    plt.ylabel(r'$Frequency$')

    plt.legend(['SuS', 'MP'])

    plt.tight_layout()
    if savepdf:
        plt.savefig('skewness_study_example_1_pf'+pf+'_hist.pdf', format='pdf', dpi=50, bbox_inches='tight')

    ii = ii + 1
plt.show()
