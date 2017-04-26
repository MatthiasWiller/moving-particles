"""
Author: Matthias Willer 2017
"""

import numpy as np

import algorithms.modified_metropolis as mmh

# p0: conditional failure probability
# N: number of samples
# target_PDF
# proposal_PDF
# LSF: limit state function g(x)

def subsetsim(p_0, N, target_PDF, proposal_PDF, LSF):
  theta = np.zeros((N), float)

  # set j = 0, number of conditional level
  j = 0 

  # set N_F(j) = 0, number o failure samples at level j
  N_F = np.zeros(N)
  g = np.zeros((N), float)
  b = np.zeros((N), float)

  # sample initial state (MCS)
  for i in range(0,N):
    theta[j][i] = np.random.uniform(0.0, 1.0)


  # check all samples in j=0 for failure
  for i in range(1,N):
    g[i] = LSF(theta[j,i])
    if g[i] > b:
      N_F[j] += 1

  while N_F[j]/N < p_0:
    j += 1 # move to next conditional level

    # sort {g(i)} : g(i1) <= g(i2) <= ... <= g(iN)
    np.sort(g)

    # define b(j) = (g(i_(N - N*p_0) + g(i_(N - N*p0 + 1)) / 2
    index = int(N-N*p_0)
    b[j] = (g[index] + g[index+1]) /2
    
    theta_star = np.zeros(N*p_0, float)
    theta_star[0] = theta[j-1]
    for k in range(1,N*p_0):
      # generate states of Markov chain using MMA/MMH
      theta_0 = theta[j-1][N-N*p0 + k]
      a = 1
      theta_star[k] = mmh.modified_metropolis(theta[i], N, marginal_PDF, proposal_PDF, LSF, b)
    # renumber theta(j,i,k) ...
    for k in range(0,N*p_0):
      

    # count failure samples
    for i in range(1,N):
      g[i] = LSF(theta[j,i])
      if g[i] > b[j]:
        N_F[j] += 1


  # estimate of p_F
  p_F_SS = (p_0**j) * N_F[j]/N

  return p_F_SS