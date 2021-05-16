
import matplotlib.pyplot as plt
import numpy as np

a = 2
b = 3
c = 4

def f_function (x, a, b, c):
    Fx = a*x**2+b*x+c
    Fdx = 2*a*x+b
    return Fx, Fdx

def train_min(alpha, epocs, MyFunc):

    # gues the minimum
    xmin = 0
    for i in range (epocs):

        Fx, Fdx = MyFunc(xmin, a, b, c)
        # As we saw in class, we can exit the function if the Fdx is lower then a pre-set minimum

        if (Fdx > 0):       # Check sign of the NIGZERET
            xmin -= Fdx*alpha
        else:
            xmin += Fdx*alpha
    return xmin

print("Ex1: Real X with minimum value - ", (-b/(2*a)), " Estimated min after iterations - ", train_min(0.001, 10000, f_function)) 

