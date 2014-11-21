# 2-Cell coupled inhibitory Fitzhugh-Nagumo model
# Robert Capps, Georgia State University, 2014

from scipy.integrate import odeint
from numpy import arange
from pylab import figure
from pylab import show
from numpy import power
from pylab import plt

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
  Isyn = Iapp*0.15*(v2-v)

  # compute state derivatives
  vd = v - power(v, 3)/3 - w + Iapp
  wd = A*(v + B - C*w)
  
  v2d = (v2 - power(v2, 3)/3 - w2) + Isyn
  w2d = A*(v2 + B - C*w2)

  # return the state derivatives
  return [vd, wd, v2d, w2d]

state0 = [0, 0, 0, 0]
t = arange(0.0, 500.0, 0.01)

state = odeint(fitznag, state0, t)

fig = figure()
plt.plot(state[:,0],state[:,1],state[:,2],state[:,3])
fig2 = figure()
plt.plot(t,state)
plt.axis([0, 450, -2, 2.5])
show()