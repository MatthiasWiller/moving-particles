"""
Author: Matthias Willer 2017
"""

import matplotlib.pyplot as plt
import numpy as np

import utilities.util as uutil

# ---------------------------------------------------------------------------
# LOAD RESULTS FROM SIMULATIONS
# ---------------------------------------------------------------------------
direction = 'python/data/'

g_list_sus     = np.load(direction + 'sus_example_1_d10_Nspl1000_Nsim100_cs_g_list.npy')

pf_mean_sus, pf_cov_sus = uutil.get_mean_and_cov_pf_from_SUS(g_list_sus, 1000, 0.1)

print('SUS: pf =', pf_mean_sus, '| cov =', pf_cov_sus)