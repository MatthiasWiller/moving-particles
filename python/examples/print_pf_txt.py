"""
Author: Matthias Willer 2017
"""

import numpy as np

import utilities.util as uutil

np.random.seed(0)
Nb = 5
p0 = 0.1
N_sus = [1100]
N_mp = [100, 50, 25]

direction = 'python/data/example5/fixed_ncall_data/'

g_list_sus     = np.load(direction + 'sus_au_beck_N' + repr(N_sus[0]) + '_Nsim100_cs_g_list.npy')
g_list_mp     = np.load(direction + 'mp_au_beck_N' + repr(N_mp[0]) + '_Nsim100_b5_cs_sss2_g_list.npy')

pf_array_sus = uutil.get_pf_array_from_SUS(g_list_sus, N_sus[0], p0)
pf_array_mp  = uutil.get_pf_array_from_MP(g_list_mp, N_mp[0])

np.savetxt(r'pf_sus.txt', pf_array_sus, newline="\n")
np.savetxt(r'pf_mp.txt', pf_array_mp, newline='\n')
