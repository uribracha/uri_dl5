import matplotlib.pyplot as plt
import numpy as np
import unit10.b_utils as u10
import random
random.seed(1)
X, Y = u10. load_dataB1W3Ex1()
plt.plot(X,Y,'b.')
def price(x,y):
    line=(max(x),min(x))

    alist=[]
    blist=[]
    for _ in range(10):
      
        alist.append(random.randrange(-20,20))
        blist.append(random.randrange(-20,20))

    for i in range(len(alist)):
        plt.plot(line,[line[0]*alist[i]+blist[i],line[1]*alist[i]+blist[i]])
    plt.show()

price(X,Y)