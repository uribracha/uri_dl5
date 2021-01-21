
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import numpy as np
import unit10.b_utils as u10

import random
random.seed(1)

X, Y = u10. load_dataB1W3Ex2()
#fig = plt.figure()
#ax = plt.axes(projection="3d")
#ax.scatter3D(X[0], X[1], Y)
#plt.show()

aLst = []
bLst = []
cLst = []
dLst = []

for i in range(1000):
    aLst.append(random.randrange(-10,10))
    bLst.append(random.randrange(-10,10))
    cLst.append(random.randrange(-10,10))
    dLst.append(random.randrange(-10,10))

## Calculate the best cost function

def y_function(x1, x2, a, b, c, d):
    return a*(x1**2) + b*x1*x2 + c*(x2**2) + d 



for l in range (len(aLst)): # run over all the potential a,b,c,d
    
    sum = 0.0
    for i in range (len(Y)):        # For each sample
        LI = y_function(X[0][i] ,X[1][i], aLst[l], bLst[l], cLst[l], dLst[l])- Y[i]     # Error for sample i
        sum += (LI)**2
    J = sum/len(Y)

    if (l ==0):
        min = J
        indexMin = l
    elif (min>J):
        min = J
        indexMin = l

print ("a={0}, b={1}, c={2}, d={3},  cost={4}".format(aLst[indexMin], bLst[indexMin], cLst[indexMin], dLst[indexMin], min))


fig = plt.figure()
ax = plt.axes(projection="3d")
X1, X2 = np.meshgrid(np.linspace(-15, 15, 30), np.linspace(-15, 15, 30))
Ywire = y_function(X1, X2, aLst[indexMin], bLst[indexMin], cLst[indexMin], dLst[indexMin])
ax.plot_wireframe(X1, X2, Ywire, color='orange')
ax.scatter3D(X[0], X[1], Y);
plt.show()
