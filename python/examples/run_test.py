"""
Author: Matthias Willer 2017
"""

import time as timer

import numpy as np


np.random.seed(0)

direction = 'python/data/example4/fixed_ncall_data/'

g_list = np.load(direction + 'mcs_liebscher_N1000000_g_list.npy')

pf_mcs = sum(g<0 for g in g_list) / len(g_list)

print(pf_mcs)