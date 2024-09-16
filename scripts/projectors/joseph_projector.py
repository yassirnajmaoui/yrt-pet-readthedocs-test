#!/usr/bin/env python

FLAG_PLOT = True

import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
if FLAG_PLOT:
    from matplotlib.ticker import MultipleLocator
    import matplotlib as mpl
#    mpl.use('Qt5Agg')
    mpl.use('TkAgg')

@dataclass
class Volume:
	nx: int
	ny: int
	nz: int
	length_x: float
	length_y: float
	length_z: float

@dataclass
class Plane:
    nx: int
    ny: int
    length_x: float
    length_y: float

def plotpoint(pt, lbl=None):
    plt.scatter([pt[0]], [pt[1]], label=lbl)

def plotline(s, r, lbl=None, length=30, color=None):
    p1 = np.array(s)
    p2 = p1+np.array(r)*length
    plt.plot([p1[0],p2[0]],[p1[1],p2[1]], label=lbl, color=color)

NB_VOXELS_COLLISIONS = 5

p = np.zeros([NB_VOXELS_COLLISIONS,3])
v1 = np.zeros([NB_VOXELS_COLLISIONS,3])
v2 = np.zeros([NB_VOXELS_COLLISIONS,3])
v3 = np.zeros([NB_VOXELS_COLLISIONS,3])
v4 = np.zeros([NB_VOXELS_COLLISIONS,3])
w_cl = np.zeros([NB_VOXELS_COLLISIONS,3])
w_fl = np.zeros([NB_VOXELS_COLLISIONS,3])
w1 = np.zeros([NB_VOXELS_COLLISIONS,1])
w2 = np.zeros([NB_VOXELS_COLLISIONS,1])
w3 = np.zeros([NB_VOXELS_COLLISIONS,1])
w4 = np.zeros([NB_VOXELS_COLLISIONS,1])
w = np.zeros([NB_VOXELS_COLLISIONS,1])

s = np.array([-2 , 2, 0])
r = np.array([2.5, 2.3, 0])

m = np.argmax(r)
rm = r[m]
r_tld = r/rm

# o is distance between s and first plane
o = -1*s[m]


for i in np.arange(NB_VOXELS_COLLISIONS):
    p[i] = (s+o*r_tld)+i*r_tld #Eq 4
    v1[i] = [np.floor(p[i][0]), np.floor(p[i][1]), np.floor(p[i][2])] # Eq 5
    v2[i] = [np.floor(p[i][0]), np.ceil(p[i][1]), np.ceil(p[i][2])]   # Eq 5
    v3[i] = [np.ceil(p[i][0]), np.floor(p[i][1]), np.ceil(p[i][2])]   # Eq 5
    v4[i] = [np.ceil(p[i][0]), np.ceil(p[i][1]), np.floor(p[i][2])]   # Eq 5

    w_cl[i] = p[i]-np.floor(p[i])   # Eq 6
    w_fl[i] = 1-w_cl[i]             # Eq 6

    #w_cl[i] = w_cl[i]/w_cl[i][m]    # Eq 7
    #w_fl[i] = w_fl[i]/w_fl[i][m]    # Eq 7

    w1[i] = w_fl[i][0]*w_fl[i][1]*w_fl[i][2] # Eq 8
    w2[i] = w_fl[i][0]*w_cl[i][1]*w_cl[i][2] # Eq 8
    w3[i] = w_cl[i][0]*w_fl[i][1]*w_cl[i][2] # Eq 8
    w4[i] = w_cl[i][0]*w_cl[i][1]*w_fl[i][2] # Eq 8
    w[i] = w1[i]+w2[i]+w3[i]+w4[i] # Weight of i-th collision



#The actual Projection
#d = s+r*3
norm_r_tld = np.sqrt(r_tld.dot(r_tld)) # The norm of the vector r_td (r tilde)
voxel_volume = 1 # This is the value at the i-th voxel, it should be an array
projection_d = norm_r_tld*np.sum(voxel_volume*w)


#Displaying:
for i in np.arange(NB_VOXELS_COLLISIONS):
    print("\n\nVoxel n# " + str(i))
    print("p: " + str(p[i]))
    print("\n\tv:")
    print("\t\tv1: " + str(v1[i]))
    print("\t\tv2: " + str(v2[i]))
    print("\t\tv3: " + str(v3[i]))
    print("\t\tv4: " + str(v4[i]))
    print("\tw_cl: " + str(w_cl[i]))
    print("\tw_fl: " + str(w_fl[i]))
    print("\n\tweights:")
    print("\t\tw1: " + str(w1[i]))
    print("\t\tw2: " + str(w2[i]))
    print("\t\tw3: " + str(w3[i]))
    print("\t\tw4: " + str(w4[i]))
    print("\t\tw: " + str(w[i]))


if FLAG_PLOT:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    spacing=1
    minorLocator = MultipleLocator(spacing)

    plotpoint([0,0], "Origin")
    plotpoint(s, "s")
    plotline([0,0],[1,0], lbl="X axis", length=10, color="Black")
    plotline([0,0],[0,1], lbl="Y axis", length=10, color="Black")
    plotline(s,r, lbl="Projection Line", length=10)
    for i in np.arange(NB_VOXELS_COLLISIONS):
        plotpoint(p[i], lbl=("p["+str(i)+"]"))
        plotpoint(v1[i], lbl=("v1["+str(i)+"]"))
        plotpoint(v2[i], lbl=("v2["+str(i)+"]"))
        plotpoint(v3[i], lbl=("v3["+str(i)+"]"))
        plotpoint(v4[i], lbl=("v4["+str(i)+"]"))

    ax.xaxis.set_minor_locator(minorLocator)
    ax.yaxis.set_minor_locator(minorLocator)
    ax.grid(which="minor")
    ax.grid(which="major")
    plt.legend()
    plt.show()
