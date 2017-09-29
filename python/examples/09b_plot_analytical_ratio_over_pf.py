"""
# ---------------------------------------------------------------------------
# Test function to test plots
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
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rcParams['text.usetex'] = True
   
savepdf = True

# DEFINE FUNCTIONS

# ratio
p0 = 0.1
ratio = lambda pf, Nb: ((np.log(pf)/np.log(p0))*(1-p0)+1)/(1-Nb*np.log(pf))

# parameters
pf_line = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])


# -----------------------------------------------------------------------------
# RATIO over PROBABILITY OF FAILURE
# -----------------------------------------------------------------------------

# initialization
nelements = len(pf_line)

ratio_Nb5 = np.zeros(nelements, float)
ratio_Nb20 = np.zeros(nelements, float)
ratio_const = 0.1*np.ones(nelements, float)

for i in range(0, nelements):
    ratio_Nb5[i] = ratio(pf_line[i], 5)
    ratio_Nb20[i] = ratio(pf_line[i], 20)


# plotting
fig  = plt.figure()

# plt.plot(pf_line, ratio_const, label=r'const.')

plt.plot(pf_line, ratio_Nb5, '--', label=r'$N_b = 5$', color='navy')
plt.plot(pf_line, ratio_Nb20, '--', label=r'$N_b = 20$', color='red')

# legend
plt.legend(loc='upper right')

# xaxis
plt.xscale('log')
plt.gca().invert_xaxis()
plt.xlabel(r'Probability of failure, $p_f$')

# yaxis
plt.ylim(0, 0.15)
plt.yticks([0, 0.05, 0.1, 0.15])
plt.ylabel(r'$\frac{N_{SuS}}{N_{MP}}$')

plt.tight_layout()
if savepdf:
    plt.savefig('ratio_over_pf.pdf', format='pdf', dpi=50, bbox_inches='tight')

plt.show()
