import random
import numpy as np
random.seed(5)
import time as timeit
def f_function (x, a=1.1, b=-7, c=6):
    Fx=2*a*x+b
    Fdx = 2*a*x+b
    return Fx,Fdx

def train_min_adaptive(alpha, epocs, MyFunc):

    # gues the minimum
    xmin = 2
    for i in range (epocs):
       
        Fx, Fdx = MyFunc(xmin)
        
        if (Fdx*alpha<0): 
            alpha *= 1.1    # stay same direction
        else:
            alpha *= -0.5   # change direction
        xmin +=np.abs(Fdx)*alpha
        xmin=round(xmin,acc)
      
    rFx, Fdx = MyFunc(xmin)
    
    return xmin, Fx,Fdx
  