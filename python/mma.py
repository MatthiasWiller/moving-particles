"""
Author: Matthias Willer 2017
"""

import numpy as np

# theta0: inital state of a Markov chain
# N: total number of states, i.e. samples
# target_PDF
# proposal_PDF

def mma(theta0, N, marginal_PDF, proposal_PDF):
  # get dimension
  d = 1

  # initialize theta
  theta = np.zeros((d,N),float)

  for i in range(1,N-1):
    # generate a candidate state xi:
    for k in range(1,d):
      # sample xi from proposal_PDF
      xi = proposal_PDF()

      # compute acceptance ratio
      r = min(marginal_PDF(xi) / marginal_PDF(theta[i]),1)

      # accept or reject xi by setting ...
      if np.uniform(0,1) < r:
        # accept
        xi = xi
      else:
        # reject
        xi = theta[i]
    
    # check whether xi E F by system analysis and accept or reject xi by setting
    if LSF(xi) < 0:
      # in failure domain -> accept
      theta[i+1] = xi
    else:
      # not in failure domain ->reject
      theta[i+1] = theta[i]
  
  # output
  return theta
