#***************************************************************************   
# Subset simulation example
# Created on Mon Feb 27 14:49:34 2017
# @author: felipeuribe
#***************************************************************************
# Based on:
# 1."MCMC algorithms for subset simulation"
#    Papaioannou et al.
#    Probabilistic Engineering Mechanics 41 (2015) 83-103.
#***************************************************************************

#  Initial import
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import ticker
    
from CS import CS
from SuS import SuS

#***************************************************************************
#  Main program
#***************************************************************************

# parameters
d  = 10      # number of dimensions  
N  = 1000    # Total number of samples for each level
p0 = 0.1     # Probability of each subset, chosen adaptively

# limit-state function
beta = 3.5
G    = lambda u: -u.sum(axis=0)/np.sqrt(d) + beta  
    
# run SuS
u_samples = SuS(G, N, d, p0)

    