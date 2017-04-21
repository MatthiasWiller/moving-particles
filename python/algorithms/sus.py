"""
Author: Matthias Willer 2017
"""

import numpy as np

# p0: conditional failure probability
# N: number of samples
# target_PDF
# proposal_PDF
# LSF: limit state function g(x)

def subsetsim(p_0, N, target_PDF, proposal_PDF, LSF):
  theta = np.zeros((N), float)
  # set j = 0, number of conditional level
  j = 0 

  # set N_F(j) = 0, number o ffailure samples at level j
  N_F = np.zeros(N)

  # check all samples in j=0 for failure
  for i in range(1,N):
    if LSF(theta[0,i] > b):
      N_F[j] += 1

  while N_F[j]/N < p_0:
    j += 1 
    # sort {g(i)} : g(i1) <= g(i2) <= ... <= g(iN)

    # define b(j) = (g(i_(N - N*p_0) + g(i_(N - N*p0 + 1)) / 2

    for k in range(1,N*p_0):
      # generate states of Markov chain using MMA/MMH
      a = 1
      
    # renumber theta(j,i,k) ...

    # count failure samples
    for i in range(1,N):
      if LSF( theta[j,i] ) > b:
        N_F[j] += 1


  # estimate of p_F
  p_F_SS = (p_0**j) * N_F[j]/N

  return p_F_SS