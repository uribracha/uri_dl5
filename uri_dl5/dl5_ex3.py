import random
def calcfunc(x):
  fx=x**3-107*x**2-9*x+3
  fdx=3*x**2-214*x-9
  return fx,fdx

def train_min(alpha, epocs, MyFunc):

    # gues the minimum
  xmin = 0
  for i in range (epocs):
    Fx,Fdx = MyFunc(xmin)
    if (Fdx>0):
      xmin -= Fdx*alpha
    else:
      xmin += Fdx*alpha
    Fx,Fdx = MyFunc(xmin)
  return xmin, Fx,Fdx

def train_max(alpha, epocs, MyFunc):

    # gues the minimum
  xmin = 0
  for i in range (epocs):
    Fx,Fdx = MyFunc(xmin)
    if (Fdx>0):
      xmin += Fdx*alpha
    else:
      xmin -= Fdx*alpha
    Fx,Fdx = MyFunc(xmin)
  return xmin,Fx, Fdx


random.seed(5) 
print("MIN INFO: ", train_min(0.001, 10000, calcfunc)) 
print("MAX INFO: ", train_max(0.001, 10000, calcfunc))