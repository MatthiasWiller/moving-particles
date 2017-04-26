"""
Author: Matthias Willer 2017
"""

import numpy as np

import time as timer

# theta0: inital state of a Markov chain
# N: total number of states, i.e. samples
# target_PDF
# proposal_PDF
# LSF: limit state function

def modified_metropolis(theta0, N, marginal_PDF, proposal_PDF, LSF,b):
  # get dimension
  d = np.size(theta0)

  # initialize theta
  theta = np.zeros((d,N), float)

  for i in range(1,N-1):
    # generate a candidate state xi:
    for k in range(1,d):
      # sample xi from proposal_PDF
      xi = proposal_PDF(theta[i-1])

      # compute acceptance ratio
      p_new = marginal_PDF(xi)
      p_old = marginal_PDF(theta[i-1])
      alpha = p_new/p_old

      # for MMH, if proposal PDFs are not symmetric (i.e. q(x,y) != q(y,x))
      q_denom = f_prop_PDF(theta[i-1], xi) # q(x,y) = g(y)
      q_numer = f_prop_PDF(xi, theta[i-1]) # q(y,x) = g(x)
      alpha = alpha * q_numer/q_denom

      r = min(alpha, 1)

      # accept or reject xi by setting ...
      if np.uniform(0, 1) <= r:
        # accept
        xi = xi
      else:
        # reject
        xi = theta[i]
    
    # check whether xi is in Failure domain system analysis and accept or reject xi
    if LSF(xi) > b:
      # in failure domain -> accept
      theta[i] = xi
    else:
      # not in failure domain ->reject
      theta[i] = theta[i-1]
  
  # output
  return theta
