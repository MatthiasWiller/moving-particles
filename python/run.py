"""
Author: Matthias Willer 2017
"""

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import sus


np.random.seed(0)

# INPUT

p_0 = 0.1   # conditional failure probability
N = 100     # number of samples

# target PDF
def target_PDF(x):
  mu = 0
  sigma = 1
  return x

# proposal PDF
def proposal_PDF(x):
  mu = 0
  sigma = 1
  return x

# Limit state function g(x)
def LSF(x):
  b = 5
  return x-b

theta = sus.subsetsim(p_0, N, target_PDF, proposal_PDF, LSF)

