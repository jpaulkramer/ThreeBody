# -*- coding: utf-8 -*-

#Packages I'm pretty sure I need

import numpy as np
import math
from scipy.spatial import distance

# Variables I know I need: and initial conditions  

m = np.array([[1],
              [2],
              [3]]) # Mass of each body.
m = m*1.989E30 # scale to mass of sun

#               x y z
s0 = np.array([[1,1,1],
               [2,2,2],
               [3,3,3]]) # Position of bodies in n dimensional space from the origin [x,y,z]in [m]

s0=s0*1.496E11 # scale to 1 AU

#               x y z
v0 = np.array([[1,1,1],
               [2,2,2],
               [3,3,3]]) # Velocity of body A in [m/s]

v0 = v0*3.0E5

#               x y z                
a0 = np.array([[1,1,1],
               [2,2,2],
               [3,3,3]]) # acceleration of body A in [m/s^2]

# Example Force Vector: 
#         A        B       C 
# F = [[[x,y,z],[x,y,z],[x,y,z]],  A
#      [[x,y,z],[x,y,z],[x,y,z]],  B
#      [[x,y,z],[x,y,z],[x,y,z]]]  C
      


# should check that inputs are correct and that we have the correct number of bodies and dimensions across

t0 = 0 # initial time in seconds for math. 

p = m*v0  # initial momentum of bodies

# initialize working variables from starting conditions
s = s0
v = v0
a = a0
numbodies = m.shape[0] # number of bodies
numdims = s.shape[1] # number of dimensions, 2 or 3

        
# relevant equations

# Force F = m*a
# Impulse dp = (F)*dt
# gravitational attraction F = G*(m1xm2)/r where r is distance between bodies and G = 6.67x10^-11 [N*m^2/kg^2]
# euclidean distance = sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2) - for 3 dimensions, works for n dimensions




# OK let's do this, 1st step, calculate distance, direction, and forces on each body

# initialize arrays to correct size
r = np.zeros((numbodies,numbodies))
f = np.zeros((numbodies,numbodies))
F = ds = np.zeros((numbodies,numbodies,numdims))
ds = np.zeros((numbodies,numbodies,numdims))
rhat = np.zeros((numbodies,numbodies,numdims))


G = 6.67E-11
M = np.zeros((numbodies,numbodies)) # separate loop for this at the beginning cause m1*m2 only needs calculated once
for i in range(numbodies): # for each body
    for j in range(numbodies): # compare to each body
        M[i,j] = m[i]*m[j]*G
        
        
for i in range(numbodies): # for each body
    for j in range(numbodies): # compare to each body
        ds[i,j] = s[j]-s[i]
        r[i,j] = np.linalg.norm(ds[i,j])
        rhat[i,j] = ds[i,j]/r[i,j] if r[i,j]!=0 else 0
        f[i,j] = (M[i,j])/(r[i,j])**2 if r[i,j]!=0 else 0
        F[i,j] = f[i,j] * rhat[i,j]
    #       print(f[i,j],'=',M[i,j],'/',r[i,j]**2)
        
#F=np.multiply(f,rhat) # Okay, why tha F doesn't this work, and the above does? something about the shape of these vectors is fucky somehow.

print('\nr-hat=\n',rhat)
print('\nr=\n',r)
print('\nf=\n',f)
print('\nF=\n',F)  



























# so bottom of code is in center of screen
