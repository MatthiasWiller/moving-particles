"""
-----------------------------------------------------------------------------
SDOF linear oscillator
-----------------------------------------------------------------------------
Created by: 
Felipe Uribe (felipe.uriber@tum.de)
Engineering Risk Analysis Group   
Technische Universitat Munchen
www.era.bgu.tum.de
-----------------------------------------------------------------------------
Version 2017-07
-----------------------------------------------------------------------------
Based on:
1."Estimation of small failure probabilities in high dimentions by SubSim"
   Siu-Kui Au & James L. Beck.
   Probabilistic Engineering Mechanics 16 (2001) 263-277
-----------------------------------------------------------------------------
"""

#  Initial import
import numpy as np  
import time as time
import matplotlib.pyplot as plt
import SDOF as sdof

#=== Definition of the white noise excitation===============================
S  = 1                          # White noise spectral intensity 
T  = 30                         # Duration of the excitation, s
dt = 0.02                       # Time increment, s
t  = np.arange(0,T+2*dt,dt)     # time instants (one more due to interpolation)
n  = len(t)-1                   # n points ~ number of random variables
# The uncertain state vector theta consists of the sequence of i.i.d.
# standard Gaussian random variables which generate the white noise input
# at the discrete time instants
W = lambda theta: np.sqrt(2*np.pi*S/dt)*theta   # random excitation
                        
#=== limit state function ==================================================
max_thresh = 1.6    # See Fig.(1) Ref.(1)
lsf = lambda theta: sdof.LSF(theta, t, W, max_thresh)

#=== do a simple Monte Carlo ===============================================
N     = int(1e3)
theta = np.random.randn(n+1,N)    # generate N samples
g     = np.zeros((N,1))           # store LSF values

tic = time.time()    
for i in range(N):
    g[i] = lsf(theta[:,i])
    if (i % 100) == 0:
        print('Iteration',i)
    
toc = time.time()-tic
Pf  = sum((g<=0)/N)
print('Elapsed time',toc,'s')
print('Failure probability P_f=',Pf)
# when computing the Pf curve it must be similar to the one in Fig[1] Ref[1]
