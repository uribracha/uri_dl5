import numpy as np
def calc_J(X,Y,a,b):
   m=Y.shape[0]
   for i in range(m):


def calca(X,Y,a,b):
    m=Y.shape[0]
    sum=2*np.sum(x)*(a*np.sum(x)+b-np.sum(y))
    return sum/m

def calcb(X,Y,a,b):
    m=y.shape[0]
    sum=2*(a*np.sum(x)+b-np.sum(Y))
    return sum/m

x=np.array([10,-3,4])
y=np.array([12,3,-4])
w=np.array([2,3])
print("cost ="+str(calc_J(x,y,w[0],w[1])))
def li(x,y,a,b):
