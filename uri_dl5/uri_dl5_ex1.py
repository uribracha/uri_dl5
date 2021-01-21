import matplotlib.pyplot as plt
import numpy as np
import unit10.b_utils as u10
import random
random.seed(1)
X, Y = u10. load_dataB1W3Ex1()
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
findab(X,Y)