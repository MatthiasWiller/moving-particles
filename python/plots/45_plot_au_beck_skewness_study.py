"""
# ---------------------------------------------------------------------------
# File to produce plots for example 5
# ---------------------------------------------------------------------------
# Created by:
# Matthias Willer (matthias.willer@tum.de)
# Engineering Risk Analysis Group
# Technische Universitat Munchen
# www.era.bgu.tum.de
# ---------------------------------------------------------------------------
# Version 2017-10
# ---------------------------------------------------------------------------
"""

import numpy as np
import scipy.stats as scps
import matplotlib.pyplot as plt

import matplotlib

import utilities.util as uutil

# create figure object with LaTeX font
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = 22
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

print("RUN file")

# set seed for randomization
np.random.seed(0)

savepdf = False

# ---------------------------------------------------------------------------
# STANDARD INPUT
# ---------------------------------------------------------------------------

# parameters
Nsus = 1100      # SUS: number of samples per conditional level
p0   = 0.1       # SUS: Probability of each subset, chosen adaptively

Nmp  = 100       # MP: Number of initial samples 

pf_ref = 5.1e-7
# pf_list = ['1e-12']
minmax_list = [-7.5,-5.5]

def ecdf(data):
    n = len(data) # Number of data points: n
    x = np.sort(data) # x-data for the ECDF: x
    y = np.arange(1, n+1) / n # y-data for the ECDF: y

    return x, y

# ---------------------------------------------------------------------------
# LOAD RESULTS FROM SIMULATIONS
# ---------------------------------------------------------------------------
direction = 'python/data/example5/fixed_ncall_data/'

g_list_sus = np.load(direction + 'sus_au_beck_N1100_Nsim100_cs_g_list.npy')
g_list_mp = np.load(direction + 'mp_au_beck_N100_Nsim100_b5_cs_sss2_g_list.npy')


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
    plt.savefig('skewness_study_au_beck_cdf.pdf', format='pdf', dpi=50, bbox_inches='tight')


fig = plt.figure()
y_pdf_sus = kde_sus(x_line_pdf)
y_pdf_mp = kde_mp(x_line_pdf)
data = np.vstack([y_pdf_sus, y_pdf_mp]).T

plt.hist([pf_array_sus, pf_array_mp], \
        bins=np.logspace(minmax_list[0], minmax_list[1], 20), \
        color=['C2', 'C1'], \
        label=['SuS', 'MP'])
plt.xscale('log')

# plt.plot(x_line_pdf, y_pdf_mp, color='C1', label='MP')
# plt.plot(x_line_pdf, y_pdf_sus, color='C2', label='SuS')


plt.xlabel(r'$p_f$')
plt.ylabel(r'Frequency')

plt.legend(['SuS', 'MP'])

plt.tight_layout()
if savepdf:
    plt.savefig('skewness_study_au_beck_hist.pdf', format='pdf', dpi=50, bbox_inches='tight')

plt.show()
