# 2-Cell coupled inhibitory Fitzhugh-Nagumo model
# Robert Capps, Georgia State University, 2014

from scipy.integrate import odeint; 
from numpy import arange;
from pylab import figure
from pylab import show
from numpy import power
import matplotlib.pyplot as plt;
import matplotlib.patches as mpatches
import numpy as np

###########################

plt.close('all')

def fitznag(state,t):
  # unpack the state vector
  v = state[0]
  w = state[1]
  v2 = state[2]
  w2 = state[3]

  # these are our constants
  A = 0.08
  B = 0.7
  C = 0.8
  Iapp = 1.2
 # Isyn = Iapp*-0.13*(v-v2)

  # compute state derivatives
  vd = v - power(v, 3)/3 - w + Iapp
  wd = A*(v + B - C*w)
  
  v2d = (v2 - power(v2, 3)/3 - w2) + Iapp - (0.02*(v))
  w2d = (A*(v2 + B - C*w2))

  # return the state derivatives
  return [vd, wd, v2d, w2d]

state0 = [0, 0, 0, 0]
t = arange(0.0, 1500.0, 0.01)

state = odeint(fitznag, state0, t)


############################ Find Phase Lag ###########################

# Voltage vectors for cell 1 and 2
v_1 = np.around(state[:,0],7)
v_2 = np.around(state[:,2],7)

np.savetxt('v_1.txt', v_1)
np.savetxt('v_2.txt', v_2)

# Find the times when Cell 1 spikes
    #pks = np.amax(v_1[0:1500000:1])
locs = np.where((v_1 >= 1.800042300000000095e+00) & (v_1 <= 1.8005))
# Find the times when Cell 2 spikes
    #pks2 = np.amax(v_2[90:100:1])
locs2 = np.where((v_2 >= 1.800042300000000095e+00) & (v_2 <= 1.8005) )

floc2 = np.transpose(locs2[0:len(locs)])
floc1 = np.transpose(locs[0:len(locs)])

floc2L = len(floc2)
floc1L = len(floc1)

# Find where the last useful spike occurs
if floc2L > floc1L:
    print('changed!')
    floc2f = floc2.copy()
    floc2f.resize(floc1L,1)
    floc1f = floc1
else:
        print('changed!!')
        floc1f = floc1.copy()
        floc1f.resize(floc2L,1)
        floc2f = floc2
np.savetxt('floc1.txt', floc1)
np.savetxt('floc2.txt', floc2)

# Tau
tau = abs(floc2f - floc1f)
np.savetxt('Tau.txt', tau)

# Period

even, odd = floc1f[::2], floc1f[1::2]
period = np.transpose(abs(even - odd))
#period = np.array([abs(i-j) for i, j in zip(floc1f[0][0::2], floc1f[0][1::2])])
np.savetxt('Period.txt', period)

tausize = len(tau)
periodf = period.copy()
periodf.resize(1,tausize)
# Find the Phase lag

Phi12 = ((tau)/(periodf))
Phif = np.transpose(Phi12)
n = arange(0.0,len(np.transpose(Phi12)),1)
np.savetxt('Phi.txt', Phi12)

# Voltage, Slow Variable Plots
fig = figure()
plt.plot(state[:,0],state[:,1],state[:,2],state[:,3])
green = mpatches.Patch(color='green',label='cell 1')
blue = mpatches.Patch(color='blue',label='cell 2')
plt.legend(handles=[green, blue])
plt.savefig('fig1.png')

fig2 = figure()
plt.plot(t,state[:,0])
plt.plot(t,state[:,2])
plt.axis([1350, 1500, -2, 2.5])
plt.legend(handles=[green, blue])
plt.savefig('fig2.png')

fig3 = figure()
plt.plot(t,state[:,1])
plt.plot(t,state[:,3])
plt.axis([0, 1500, -2, 2.5])
plt.legend(handles=[green, blue])
plt.savefig('fig3.png')

fig4 = figure()
plt.plot(n,Phif)
plt.autoscale(enable=True,tight=True)

# Phase Plot
print("The final phase lag is " +repr(Phif))
show()