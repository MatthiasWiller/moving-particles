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
matplotlib.rcParams.update({'font.size': 23})
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rcParams['text.usetex'] = True
   

# DEFINE FUNCTIONS

# Monte Carlo
monte_carlo = lambda pf, delta2: (1-pf)/(pf*delta2)

# Subset Simulation
p0 = 0.1
subset = lambda pf, delta2, gamma: ((-np.log10(pf/p0)*(1-p0)/np.sqrt(p0))**2 - np.log10(pf/p0)*(1-p0))*((1+gamma)/(delta2))

# Moving particles
moving_particles = lambda pf, delta2, burnin: ((burnin + 1)*(-np.log(pf))+1)*(-np.log(pf))/(np.log(delta2 + 1))

# parameters
pf_line = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
delta2_fixed = 0.001

pf_fixed = 1e-6
delta_line = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
delta2_line = delta_line*delta_line

gamma_min = 0
gamma_max = 10

burnin_min = 0
burnin_max = 20


# -----------------------------------------------------------------------------
# COST over PROBABILITY OF FAILURE
# -----------------------------------------------------------------------------

# initialization
nelements = len(pf_line)

cost_mc = np.zeros(nelements, float)
cost_sus_min = np.zeros(nelements, float)
cost_sus_max = np.zeros(nelements, float)
cost_mp_min = np.zeros(nelements, float)
cost_mp_max = np.zeros(nelements, float)


for i in range(0, nelements):
    cost_mc[i] = monte_carlo(pf_line[i], delta2_fixed)

    cost_sus_min[i] = subset(pf_line[i], delta2_fixed, gamma_min)
    cost_sus_max[i] = subset(pf_line[i], delta2_fixed, gamma_max)

    cost_mp_min[i] = moving_particles(pf_line[i], delta2_fixed, burnin_min)
    cost_mp_max[i] = moving_particles(pf_line[i], delta2_fixed, burnin_max)




# plotting
fig  = plt.figure()

plt.plot(pf_line, cost_mc, label=r'Monte Carlo')

# plt.plot(pf_line, cost_sus_min, '--', label=r'SuS (min)', color='navy')
# plt.plot(pf_line, cost_sus_max, '--', label=r'SuS (max)', color='navy')

plt.fill_between(pf_line, cost_sus_min, cost_sus_max, label=r'SuS', alpha=0.3, color='navy')

# plt.plot(pf_line, cost_mp_min, '--', label=r'MP (min)', color='red')
# plt.plot(pf_line, cost_mp_max, '--', label=r'MP (max)', color='red')

plt.fill_between(pf_line, cost_mp_min, cost_mp_max, label=r'MP', alpha=0.3, color='red')

# legend
plt.legend()

# xaxis
plt.xscale('log')
plt.gca().invert_xaxis()
plt.xlabel(r'Probability of failure, $p_f$')

# yaxis
plt.yscale('log')
plt.ylabel(r'Computational cost, $t$')

plt.tight_layout()
plt.savefig('comp_time_pf.pdf', format='pdf', dpi=50, bbox_inches='tight')



# -----------------------------------------------------------------------------
# COST OVER C.O.V.
# -----------------------------------------------------------------------------

# initialization
nelements = len(delta2_line)

cost_mc = np.zeros(nelements, float)
cost_sus_min = np.zeros(nelements, float)
cost_sus_max = np.zeros(nelements, float)
cost_mp_min = np.zeros(nelements, float)
cost_mp_max = np.zeros(nelements, float)

# evaluate functions
for i in range(0, nelements):
    cost_mc[i] = monte_carlo(pf_fixed, delta2_line[i])
    
    cost_sus_min[i] = subset(pf_fixed, delta2_line[i], gamma_min)
    cost_sus_max[i] = subset(pf_fixed, delta2_line[i], gamma_max)

    cost_mp_min[i] = moving_particles(pf_fixed, delta2_line[i], burnin_min)
    cost_mp_max[i] = moving_particles(pf_fixed, delta2_line[i], burnin_max)


# plotting
fig  = plt.figure()

plt.plot(delta2_line, cost_mc, label=r'Monte Carlo')

# plt.plot(delta2_line, cost_sus_min, '--', label=r'SuS (min)', color='navy')
# plt.plot(delta2_line, cost_sus_max, '--', label=r'SuS (max)', color='navy')

plt.fill_between(delta2_line, cost_sus_min, cost_sus_max, label=r'SuS', alpha=0.3, color='navy')

# plt.plot(delta2_line, cost_mp_min, '--', label=r'MP (min)', color='red')
# plt.plot(delta2_line, cost_mp_max, '--', label=r'MP (max)', color='red')

plt.fill_between(delta2_line, cost_mp_min, cost_mp_max, label=r'MP', alpha=0.3, color='red')

# legend
plt.legend()

# xaxis
#plt.xscale('log')
#plt.gca().invert_xaxis()
plt.xlabel(r'Coefficient of variation, $\delta^2$')

# yaxis
plt.yscale('log')
plt.ylabel(r'Computational cost, $t$')

plt.tight_layout()
plt.savefig('comp_time_cov.pdf', format='pdf', dpi=50, bbox_inches='tight')

plt.show()
