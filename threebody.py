# -*- coding: utf-8 -*-

#Packages I'm pretty sure I need
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera

# Variables I know I need: and initial conditions  

m = np.array([[1],
              [2],
              [3]]) # Mass of each body.
m = m*1.989E30 # scale to mass of sun

#               x y z
s0 = np.array([[1,3,1],
               [2,1,2],
               [3,2,3]]) # Position of bodies in n dimensional space from the origin [x,y,z]in [m]

s0=s0*1.496E11 # scale to 1 AU

#               x y z
v0 = np.array([[1,1,1],
               [2,2,2],
               [3,3,3]]) # Velocity of body A in [m/s]

v0 = v0*3.0E5 # Scale to velocity of Earth around sun, for giggles

#               x y z                
a0 = np.array([[1,1,1],
               [2,2,2],
               [3,3,3]]) # acceleration of body A in [m/s^2]

# Scale to effectively 0 acceleration, let's see how this works, shall we?

# Example Force Vector: 
#         A        B       C 
# F = [[[x,y,z],[x,y,z],[x,y,z]],  A
#      [[x,y,z],[x,y,z],[x,y,z]],  B
#      [[x,y,z],[x,y,z],[x,y,z]]]  C

#F[:,:,0] = [A[x,x,x], B[x,x,x], C[x,x,x]]      


# should check that inputs are correct and that we have the correct number of bodies and dimensions across

t0 = 0 # initial time in days for tracking
tn = 1000 # let's simulate an earth- solar year
dt = 1 # T step in Days - will convert to seconds for math in the loop
tscale = 86400*365 # convert seconds to days

ts = np.zeros(([tn,2]))

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
# seconds in 1 day: 86400

Ax = [s[0,0]]
Bx = [s[1,0]]
Cx = [s[2,0]]
Ay = [s[0,1]]
By = [s[1,1]]
Cy = [s[2,1]]
Az = [s[0,2]]
Bz = [s[1,2]]
Cz = [s[2,2]]

# OK let's do this, 1st step, calculate distance, direction, and forces on each body

# initialize arrays to correct size
r = np.zeros((numbodies,numbodies))
f = np.zeros((numbodies,numbodies))
F = ds = np.zeros((numbodies,numbodies,numdims))
ds = np.zeros((numbodies,numbodies,numdims))
rhat = np.zeros((numbodies,numbodies,numdims))
dp = np.zeros((numbodies,numdims))

G = 6.67E-11
M = np.zeros((numbodies,numbodies)) # separate loop for this at the beginning cause m1*m2 only needs calculated once
for i in range(numbodies): # for each body
    for j in range(numbodies): # compare to each body
        M[i,j] = m[i]*m[j]*G

plt.rcParams['legend.fontsize'] = 10
fig = plt.figure()        
ax = fig.gca(projection='3d')
camera = Camera(fig)
        
# Main Loop        
for t in range(0, tn, dt):
    
    # move the bodies at their velocities
    
    
    for i in range(numbodies): # for each body
        for j in range(numbodies): # compare to each body
            ds[i,j] = s[j]-s[i]
            r[i,j] = np.linalg.norm(ds[i,j])
            rhat[i,j] = ds[i,j]/r[i,j] if r[i,j]!=0 else 0
            f[i,j] = (M[i,j])/(r[i,j])**2 if r[i,j]!=0 else 0
            F[i,j] = f[i,j] * rhat[i,j]
    
        #   print(f[i,j],'=',M[i,j],'/',r[i,j]**2)
        
        #F=np.multiply(f,rhat) # Okay, why tha F doesn't this work, and the above does? something about the shape of these vectors is fucky somehow.

    # summarize Forces into simple x, y, z components for each body
    sumFx = np.sum(F[:,:,0],1) # = [Ax, Bx, Cx]
    sumFy = np.sum(F[:,:,1],1) # = [Ay, By, Cy]
    sumFz = np.sum(F[:,:,2],1) # = [Az, Bz, Cz]
    sumF = np.transpose(np.array([sumFx,sumFy,sumFz]))

    # OK, now let's figure out how the momentum of our bodies is changing. 
    
    dp = sumF*(dt*tscale) # impulse
    #dv = dp/m # measure acceleration
    p = p+dp # update momentum
    v = p/m # update velocities from momentum
    s = s+v # move the bodies by their velocities
    
    Ax.append(s[0,0])
    Bx.append(s[1,0])
    Cx.append(s[2,0])
    
    Ay.append(s[0,1])
    By.append(s[1,1])
    Cy.append(s[2,1])
    
    Az.append(s[0,2])
    Bz.append(s[1,2])
    Cz.append(s[2,2])
    
    ax.plot([s[0,0]], [s[0,1]], [s[0,2]], color='red') # body A
    ax.plot([s[1,0]], [s[1,1]], [s[1,2]], color='green') # body B
    ax.plot([s[2,0]], [s[2,1]], [s[2,2]], color='blue') # body C
    camera.snap()
    
#anim = camera.animate(blit=False, interval=10)
#anim.save('3d.mp4')

ax.plot(Ax, Ay, Az, color='red') # body A
ax.plot(Bx, By, Bz, color='green') # body A
ax.plot(Cx, Cy, Cz, color='blue') # body A
#ax.plot([s[1,0]], [s[1,1]], [s[1,2]], color='green') # body B
#ax.plot([s[2,0]], [s[2,1]], [s[2,2]], color='blue') # body C

# print('\nr-hat=\n',rhat)
# print('\nr=\n',r)
# print('\nf=\n',f)
# print('\nF=\n',F)  



























# so bottom of code is in center of screen
