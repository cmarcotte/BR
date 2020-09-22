# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:08:08 2020

@author: eisne
"""
import numpy as np
import matplotlib.pyplot as plt
#from IPython import get_ipython
from scipy.integrate import solve_ivp
from itertools import chain
import random



def ab(C, V):
    
	# eq (13) from original paper
	return (C[0]*np.exp(C[1]*(V+C[2]))+C[3]*(V+C[4]))/(np.exp(C[5]*(V+C[2]))+C[6])


p = [0.05, 2.3, 3.25]
def BR(t, y, c, a, f):
    
	# following https://doi.org/10.1016/0167-2789(84)90283-5

	# p = [C, A, f]
	
	# Harmonic oscillator variables y, z are last#								     v  v
 # x = [V, Ca, x, m, h, j, d, f, y, z]

    dy = np.zeros(np.shape(y))
        
    V = y[0]
    

		# these from Beeler & Reuter table:
#"""
    ax = ab([ 0.0005, 0.083, 50.0, 0.0, 0.0, 0.057, 1.0],V)
    bx = ab([ 0.0013,-0.06 , 20.0, 0.0, 0.0,-0.04 , 1.0],V)
    am = ab([ 0.0   , 0.0  , 47.0,-1.0,47.0,-0.1  ,-1.0],V)
    bm = ab([40.0   ,-0.056, 72.0, 0.0, 0.0, 0.0  , 0.0],V)
    ah = ab([ 0.126 ,-0.25 , 77.0, 0.0, 0.0, 0.0  , 0.0],V)
    bh = ab([ 1.7   , 0.0  , 22.5, 0.0, 0.0,-0.082, 1.0],V)
    aj = ab([ 0.055 ,-0.25 , 78.0, 0.0, 0.0,-0.2  , 1.0],V)
    bj = ab([ 0.3   , 0.0  , 32.0, 0.0, 0.0,-0.1  , 1.0],V)
    ad = ab([ 0.095 ,-0.01 , -5.0, 0.0, 0.0,-0.072, 1.0],V)
    bd = ab([ 0.07  ,-0.017, 44.0, 0.0, 0.0, 0.05 , 1.0],V)
    af = ab([ 0.012 ,-0.008, 28.0, 0.0, 0.0, 0.15 , 1.0],V)
    bf = ab([ 0.0065,-0.02 , 30.0, 0.0, 0.0,-0.2  , 1.0],V)
	#"""
	
    IK=(np.exp(0.08*(V + 53.0)) + np.exp(0.04*(V + 53.0)))
    IK=4.0*(np.exp(0.04*(V + 85.0)) - 1.0)/IK
    IK=IK + 0.2*(V + 23.0)/(1.0-np.exp(-0.04*(V + 23.0)))
    IK=0.35*IK
    Ix = y[2]*0.8*(np.exp(0.04*(V + 77.0))-1.0)/np.exp(0.04*(V + 35.0))
    INa = (4.0*y[3]*y[3]*y[3]*y[4]*y[5] + 0.003)*(V - 50.0)
    Is = 0.09*y[6]*y[7]*(V + 82.3 + 13.0287*np.log(y[1]))
	
# this is the forcing current

# BR dynamics
    
    I = a*(y[8]**500)

    dy[0] = (I - IK - Ix - INa - Is)/c
    dy[1] = (-10e-7 * Is + 0.07*(10e-7 - y[1]))
    dy[2] = (ax*(1.0 - y[2]) - bx*y[2])
    dy[3] = (am*(1.0 - y[3]) - bm*y[3])
    dy[4] = (ah*(1.0 - y[4]) - bh*y[4])
    dy[5] = (aj*(1.0 - y[5]) - bj*y[5])
    dy[6] = (ad*(1.0 - y[6]) - bd*y[6])
    dy[7] = (af*(1.0 - y[7]) - bf*y[7])
	
# nonlinear oscillator dynamics
    ω = 2*np.pi*f/1000.0
    dy[8] = (ω*y[9] + y[8] *(1.0 - y[8]**2 - y[9]**2))
    dy[9]=(-ω*y[8]  + y[9]*(1.0 - y[8]**2 - y[9]**2))
    

        

    return dy
	
x0 = np.array([-84.0, 10e-7, 0.01, 0.01, 0.99, 0.99, 0.01, 0.99, 0.0, 1.0])
t_span = (0, 500)


V_thresh = -75

APDs = []
DIs = []
CLs = []

for f_it in range(200, 12, -1):
    f = f_it*0.1
    APD = []
    DI = []
    
    print("Progress: " + str(100*(200 - f_it)/200) + '%')
    sol = solve_ivp(BR, t_span, x0, method='LSODA', args=(0.05, 10, f), dense_output=True)
    t = sol.t
    V = sol.sol(t)[0]
    
    for i in range(len(V)):
        
        if np.sign(V[i -1] - V_thresh) < np.sign(V[i] - V_thresh):
            APD.append(sol.t[i])
        elif np.sign(V[i -1] - V_thresh) > np.sign(V[i] - V_thresh):
            DI.append(sol.t[i])
            
    m = np.min([len(APD) - 1, len(DI)])
    
    
    APD = np.array(APD)
    DI = np.array(DI)
    CLs.append([1000/f]*(len(DI) - 1))
    APDs.append(DI[:m] - APD[:m])
    DIs.append(APD[1:m +1] - DI[:m])



APDs = list(chain.from_iterable(APDs))
DIs = list(chain.from_iterable(DIs))
CLs = list(chain.from_iterable(CLs))

print(len(CLs))
plt.figure(2)
plt.scatter(CLs, APDs)
plt.show()
    
    
    
    


