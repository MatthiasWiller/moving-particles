"""
-----------------------------------------------------------------------------
4th order Runge-Kutta to solve ODEs
-----------------------------------------------------------------------------
Created by: 
Felipe Uribe (felipe.uriber@tum.de)
Engineering Risk Analysis Group   
Technische Universitat Munchen
www.era.bgu.tum.de
-----------------------------------------------------------------------------
Version 2017-07
-----------------------------------------------------------------------------
Check also: from scipy.integrate import ode
-----------------------------------------------------------------------------
"""
#  Initial import
import numpy as np
import scipy as sp
from scipy import interpolate
# import matplotlib.pyplot as plt

#=============================================================================
def LSF(theta, t, W, max_thresh):
    
    # compute the white noise and create an interpolant
    Wt    = np.squeeze(W(theta))
    Xdd_g = sp.interpolate.interp1d(t, Wt, kind='linear')
    # plt.plot(t,Wt)  plt.plot(t,Xdd_g(t),'r--')   # must coincide
    
    # some constants
    t = t[:-1]        # delete the additional time after interpolation
    h = t[1]-t[0]     # time step
    n = len(t)        # number of times (random variables)

    # define SDOF differential equation
    omega = 7.85            # Natural frequency, rad/s
    zeta  = 0.02            # Damping ratio
    diff_eq = lambda t, X: np.asarray([ X[1], \
                                       -Xdd_g(t) -2*zeta*omega*X[1] - omega**2*X[0] ])  
    
    # apply RK4 to solve the ODE
    x      = np.zeros((n,2))     # store the solution (displacement,velocity,acceleration)
    x[0,:] = np.zeros((1,2))     # initial state (displ,vel)
    for k in range(n-1):
        k1       = diff_eq(t[k], x[k,:])
        k2       = diff_eq(t[k] + h/2, x[k,:] + h/2*k1)
        k3       = diff_eq(t[k] + h/2, x[k,:] + h/2*k2)
        k4       = diff_eq(t[k] + h, x[k,:]+ h*k3)
        x[k+1,:] = x[k,:] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    # solution of the ODE
    dis = x[:,0]
    #vel = x[:,1]
    #acc = -Xdd_g(t) - 2*zeta*omega*vel - omega**2*dis 
                
    # compute the LSF value
    g = max_thresh - max(abs(dis))

    # plots
#==============================================================================
#     f, axarr = plt.subplots(3, sharex=True)
#     axarr[0].plot(t,dis,'k-')
#     axarr[0].set_ylabel('Displacement')
#     axarr[0].set_xlim([0,max(t)])
#     axarr[1].plot(t,vel,'r-')  # velocity
#     axarr[1].set_ylabel('Velocity') 
#     axarr[1].set_xlim([0,max(t)])
#     axarr[2].plot(t,acc,'b-')  # velocity
#     axarr[2].set_ylabel('Acceleration') 
#     axarr[2].set_xlim([0,max(t)])
#==============================================================================
        
    return g