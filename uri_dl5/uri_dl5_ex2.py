import matplotlib.pyplot as plt
import numpy as np
import unit10.b_utils as u10
import random
from mpl_toolkits import mplot3d
random.seed(1)
''' X, Y = u10. load_dataB1W3Ex1()
plt.plot(X,Y,'b.')
def plot(x,y):
    line=(max(x),min(x))
    alist=[]
    blist=[]
   for _ in range(10):
      
        alist.append(random.randrange(-20,20))
        blist.append(random.randrange(-20,20))
    for i in range(len(alist)):
        plt.plot(line,[line[0]*alist[i]+blist[i],line[1]*alist[i]+blist[i]])
    plt.show()
   
   
def findab(x,y):
    alist=[]
    blist=[]
    for _ in range(10):
        alist.append(random.randrange(-20,20))
        blist.append(random.randrange(-20,20))
       
    min=0
    indexmin=-1
   
    for i in range(len(alist)):
        sum=0
        for j in range(len(x)):
            sum+=(x[j]*alist[i]+blist[i]-y[j])**2
        z=sum/len(x)
        if (indexmin==-1):
        min=z
        indexmin=i
        elif (min>z):
            min=z
            indexmin=i

    print("a={0} , b={1} , cost={2}".format(alist[indexmin],blist[indexmin],min))
    '''


def calc(x1,x2,y):
    a=[]
    b=[]
    c=[]
    d=[]
    index=-1
    min=0
    
    for _ in range(1000):
        
        a.append(random.randrange(-10,10))
        b.append(random.randrange(-10,10))
        c.append(random.randrange(-10,10))
        d.append(random.randrange(-10,10))
        for i in range(len(a)):
            sums=0
            for j in range(len(x1)):
                sums+=(a[i]*x1[j]+b[i]*(x1[j]*x2[j])+(c[i]*(x1[j]**2)+d[i]))**2-(y[j])**2
        z =sums/len(x1)
        if(index==-1):
            index=i
            min=z
        elif(z<min):
            index=i
            min=z
    print("a={0},b={1},c={2},d={3},cost={4}". format(a[index],b[index],c[index],d[index],min))
X1, Y1 = u10. load_dataB1W3Ex2()
print(X1,Y1)
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(X1[0], X1[1], Y1)
plt.show()

calc(X1[0], X1[1], Y1)