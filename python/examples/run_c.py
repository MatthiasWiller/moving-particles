"""
Author: Matthias Willer 2017
"""

import numpy as np

np.random.seed(0)
direction = 'python/data/example5/fixed_ncall_data/'

mcs_data = np.loadtxt(direction + 'mcs_au_beck_N10000000.txt', delimiter=',')

b_line_mcs = mcs_data[:,0]
pf_line_mcs = mcs_data[:,1]
N = len(b_line_mcs)
Nf = 0
for g in b_line_mcs:
    if g < 0:
        Nf = Nf + 1

print(Nf/N)
