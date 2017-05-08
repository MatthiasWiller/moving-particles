"""
Author: Matthias Willer 2017
"""

import numpy as np

import algorithms.modified_metropolis as mmh

# p0: conditional failure probability
# n_samples_per_level: number of samples per conditional level
# target_PDF
# proposal_PDF
# LSF: limit state function g(x)

def subsetsim(p0, n_samples_per_level, d, marginal_PDF, sample_prop_PDF, f_prop_PDF, LSF):
  # initialization and constants
  #theta = np.zeros((N), float)
  theta = []
  max_it = 20
  #N_fail = np.zeros((max_it, 1), int)
  Nf = np.zeros(max_it)
  #b = np.zeros((max_it, 1), float)
  b = np.zeros(max_it)

  n_seeds_per_level = int(n_samples_per_level*p0)
  n_samples_per_seed = int(1/p0) # including the seed 
  

  # set N_F(j) = 0, number o failure samples at level j
  g = []
  #g = np.zeros((N), float)

  # sample initial step (MCS)
  j = 0 # set j = 0, number of conditional level
  #N_fail[j] = 0
  theta0 = np.random.randn(n_samples_per_level, d)
  g0 = np.zeros((n_samples_per_level), float)
  for i in range(0, n_samples_per_level):
    #theta[j][i] = np.random.randn(d, n_samples_per_level)
    g0[i] = LSF(theta0[i,:])
    if (g0[i] <= 0):
      Nf[j] += 1
  
  theta.append(theta0)
  g.append(g0)

  # Subset Simulation steps
  while Nf[j] < n_seeds_per_level:
    j += 1 # move to next conditional level

    #g_prime = zip(theta0, g0)
    #g_prime.sorted(key=g0)

    # sort {g(i)} : g(i1) <= g(i2) <= ... <= g(iN)
    g_prime = np.sort(g0) # sorted g
    idx = sorted(range(len(g0)), key=lambda x: g0[x])

    # order samples according to the previous order
    theta_prime = theta0[(idx)] # sorted theta

    # compute intermediate threshold level
    # define b(j) = (g(i_(N - N*p_0) + g(i_(N - N*p0 + 1)) / 2
    b[j] = (g_prime[n_samples_per_level- n_seeds_per_level] + g_prime[n_samples_per_level - n_seeds_per_level + 1]) /2
    print("\nIntermediate threshold level b=", b[j], "\n")
    
    # select seeds for the MCMC sampler
    theta_tilde = theta_prime[0:n_samples_per_seed]

    #theta_star = np.zeros(n_seeds_per_level, float)
    #theta_star[0] = theta[j-1]

    theta_star = []
    for k in range(0, n_seeds_per_level):
      # generate states of Markov chain using MMA/MMH
      #theta_0 = theta[j-1][N-N*p0 + k]
      #a = 1
      theta_temp = mmh.modified_metropolis(theta_tilde[k, :], n_samples_per_seed, marginal_PDF, sample_prop_PDF, f_prop_PDF, LSF, b[j])
      theta_star.append(theta_temp)
    

    theta0 = np.zeros((n_samples_per_level, d), float)
    # renumber theta(j,i,k) ...
    for k in range(0, n_seeds_per_level):
      theta0[n_samples_per_seed*(k):n_samples_per_seed*(k+1), :] = theta_star[k][:, :]
      
    theta.append(theta0)

    # count failure samples
    for i in range(0, n_samples_per_level):
      g0[i] = LSF(theta0[i, :])
      if g0[i] > b[j]:
        Nf[j] += 1
  

  # estimate of p_F
  p_F_SS = (p0**j) * Nf[j]/n_samples_per_level

  return p_F_SS, theta