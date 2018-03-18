# %% 1
# Package imports
import sys
sys.path.append('ellipsoid/')
import ellipsoid as el
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import scipy.optimize as opt
import math
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
import scipy
from numpy import exp
import itertools
import pandas as pd
from matplotlib.lines import Line2D
import pdb

# Plotting helpers
colors = ['k','b','g','r','m']
markers = []
for m in Line2D.markers:
    try:
        if len(m) == 1 and m != ' ':
            markers.append(m)
    except TypeError:
        pass


sample_size = 1001

#Make some straight line data
x = np.linspace(0, 4, sample_size)

theta0 = 1;
theta1 = 2;
# alternatively write
theta = np.zeros((2,1)) #similar to fortran, indexing starts at 1 (unlike Fortran can't change it)
theta[0] = theta0
theta[1] = theta1

y = theta0 + theta1*x
y = y + np.random.randn(len(y))*2

# Plot dataset
fig = plt.figure()
ax1 = fig.add_subplot(111)

fileLabel = "Test Data Set"
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.axis('equal')
ax1.axis([-5,5,-10,15],'equal')
for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                     ax1.get_xticklabels() + ax1.get_yticklabels()):
                item.set_fontsize(18)
# ax1.plot(x, y, color = colors[0], marker = markers[0], linestyle='None')
ax1.set_title("Noisy linear relation");

############################################################
##### SVD on Original Data gives principal axes directions##
############################################################
#Standardize data
X = [x,y]
mean_X = np.mean(X,axis=1)
Xnorm = np.asarray([x-mean_X[0],y-mean_X[1]]).T #Center the data on (0,0)

# Run SVD on data points
u, s, vh = np.linalg.svd(Xnorm,full_matrices = True)
print('s:',s)
print('vh:',vh)
smat = np.pad(np.diag(s),((0,sample_size-2),(0,0)),'constant')
# print('smat.vh',np.dot(smat,vh))
ax1.plot(Xnorm.T[0], Xnorm.T[1], color = colors[1], marker = markers[0], linestyle='None')
ax1.plot([0,vh.T[0,0]],[0,vh.T[1,0]], linestyle='-', color = colors[2])
ax1.plot([0,vh.T[0,1]],[0,vh.T[1,1]], linestyle='-', color = colors[3])

#Now you have the directions - what are the magnitudes?
# vh.T[0,0],vh.T[1,0] is a vector describing a principal axis
# X is a vector describing the position of a point
# First attempt: project points to principal axes, find abs. max.
PrincAxis1Vec = [vh.T[0,0], vh.T[1,0]]
DistVect1 = [[np.dot(Xnorm[i], PrincAxis1Vec)] for i in range(sample_size)]
Size1 = np.max(np.abs(DistVect1))

PrincAxis2Vec = [vh.T[0,1], vh.T[1,1]]
DistVect2 = [[np.dot(Xnorm[i],PrincAxis2Vec)] for i in range(sample_size)]
Size2 = np.max(np.abs(DistVect2))

norm1 = Size1/np.sqrt(vh.T[0,0]**2+vh.T[1,0]**2)
norm2 = Size2/np.sqrt(vh.T[0,1]**2+vh.T[1,1]**2)

ax1.plot([0,norm1*vh.T[0,0]],[0,norm1*vh.T[1,0]], linestyle='-.', color = colors[2])
ax1.plot([0,norm2*vh.T[0,1]],[0,norm2*vh.T[1,1]], linestyle='-.', color = colors[3])
ax1.plot([0,-norm1*vh.T[0,0]],[0,-norm1*vh.T[1,0]], linestyle='-.', color = colors[2])
ax1.plot([0,-norm2*vh.T[0,1]],[0,-norm2*vh.T[1,1]], linestyle='-.', color = colors[3])
############################################################

############################################################
##### Khachiyan Algorithm ##################################
############################################################
    # find the ellipsoid
ET = el.EllipsoidTool()
(center, radii, rotation) = ET.getMinVolEllipse(Xnorm, .01)
print(center, radii)

# fig2 = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# plot points
#    ax.scatter(P[:,0], P[:,1], P[:,2], color='g', marker='*', s=100)
# ax1.scatter(P[:,0], P[:,1], color='g', marker='*', s=100)

# plot ellipsoid
# ET.plotEllipsoid(center, radii, rotation, ax=ax1, plotAxes=True)
ET.plotEllipse(center, radii, rotation, ax=ax1, plotAxes=True)  
ET.plotEllipse([0,0],[norm1,norm2],vh, plotAxes=True)
plt.show()

