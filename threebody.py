# -*- coding: utf-8 -*-

#Packages I'm pretty sure I need

import numpy as np
import math
from scipy.spatial import distance

# Variables I know I need: and initial conditions  

m = np.array([[1],[2],[3]]) # Mass of bodies
s0 = np.array([[1,1,1],[2,2,2],[3,3,3]]) # Position of bodies in 2 dimensional plane from the origin [0,0] , [x,y,z] in [m]
v0 = np.array([[1,1,1],[2,2,2],[3,3,3]]) # Velocity of body A in [m/s]
a0 = np.array([[1,1,1],[2,2,2],[3,3,3]]) # acceleration of body A in [m/s^2] 
t0 = 0 # initial time in seconds for math. 

p = m*v0  # initial momentum of bodies

# relevant equations

# Force F = m*a
# Impulse dp = (F)*dt
# gravitational attraction F = G*(m1xm2)/r where r is distance between bodies and G = 6.67x10^-11 [N*m^2/kg^2]

G = 6.67*10^-11

# OK let's do this, 1st step.

# First calculate the distance between the bodies

r = distance.cdist(s0, s0, 'euclidean')

# second calculate the forces each of these suckers are inflicting on one another

print('m= ', m)
print('s= ', s0)
print('v= ', v0)
print('a= ', a0)

print('p= ', p)

print(np.sum(p))

print('r= ', r)