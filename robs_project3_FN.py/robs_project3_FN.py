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
  theta = -1.952
  Erev = 1.7
  Iapp = 1.2
  gsyn = 0.1
  S = 0.01
  lam = 1.0155
  # = gsyn*(-(v-Erev) * (1/(1+np.power(np.e,-k*(v2-theta)))) - (Erev/(1+np.power(np.e,(k*theta)))))
  Isyn = gsyn*((v-Erev)*S)/(1+np.power(np.e,(lam*(v2-theta)))) + 1/(1+np.power(np.e,(lam*(v2-theta))))
  #print(Isyn)
  # here are our derivatives :-)
  vd = v - power(v, 3)/3 - w + Iapp + Isyn
  wd = A*(v + B - C*w)
  
  v2d = v2 - power(v2, 3)/3 - w2 + Isyn + Iapp
  w2d = A*(v2 + B - C*w2)

  # return the state derivatives
  return [vd, wd, v2d, w2d]

state0 = [0, 0, 0, 0]
t = arange(0.0, 2800.0, 0.01)

state = odeint(fitznag, state0, t,rtol=1.49012e-10,atol=1.49012e-10)


############################ Find Phase Lag ###########################

# Voltage vectors for cell 1 and 2
v_1 = np.around(state[:,0],7)
v_2 = np.around(state[:,2],7)

np.savetxt('v_1.txt', v_1)
np.savetxt('v_2.txt', v_2)



# Find the times when Cell 1 spikes
locs = (np.diff(np.sign(np.diff(v_1))) < 0).nonzero()[0] +1

# Find the times when Cell 2 spikes
locs2 = (np.diff(np.sign(np.diff(v_2))) < 0).nonzero()[0] +1


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
        print('This thing printed.')
        floc1f = floc1.copy()
        floc1f.resize(floc2L,1)
        floc2f = floc2
np.savetxt('floc1.txt', floc1)
np.savetxt('floc2.txt', floc2)

# Tau
tau = np.around(abs(floc2f - floc1f),decimals=3)
np.savetxt('Tau.txt', tau)

# Period
even, odd = floc1f[0::2], floc1f[1::2]
evenf, oddf = floc1f[0::2], floc1f[1::2]
if len(even) > len(odd):
    evenf = even.copy()
    evenf.resize(len(odd),1)
    oddf = odd.copy()
    print('That is odd... is it not?')
if len(odd) > len(even):
    oddf = odd.copy()
    oddf.resize(len(even),1)
    evenf = even.copy()
    print('Nah, it is even!')
period = np.transpose(np.around(abs(evenf - oddf),decimals=3))
np.savetxt('Period.txt', period)

# resize our period
periodsize = len(np.transpose(period))
tauf = tau.copy()
tauf.resize(1,periodsize)

# Find the Phase lag
Phi12 = np.true_divide((tauf),(period))
Phif = np.transpose(np.arctan(Phi12)) % 1
n = arange(0,len(Phif),1)
np.savetxt('Phi.txt', Phi12)

# Voltage, Slow Variable Plots
fig = figure()
plt.plot(state[:,0],state[:,1],state[:,2],state[:,3])
green = mpatches.Patch(color='green',label='cell 2')
blue = mpatches.Patch(color='blue',label='cell 1')
plt.legend(handles=[green, blue])
plt.ylabel('V')
plt.xlabel('W')
plt.savefig('fig1.png')

fig2 = figure()
plt.plot(t,state[:,0])
plt.plot(t,state[:,2])
plt.axis([0, 2800, -2, 2.5])
plt.legend(handles=[green, blue])
plt.xlabel('time (ms)')
plt.ylabel('V')
plt.savefig('fig2.png')

fig3 = figure()
plt.plot(t,state[:,1])
plt.plot(t,state[:,3])
plt.axis([0, 2000, 0, 2.5])
plt.legend(handles=[green, blue])
plt.xlabel('time (ms)')
plt.ylabel('W')
plt.savefig('fig3.png')

# Phase plot
fig4 = figure()
plt.plot(n,Phif)
plt.autoscale(enable=True,tight=True)
plt.xlabel('# of oscillations')
plt.ylabel('Phase difference, mod(1)')
plt.savefig('fig4.png')
show()