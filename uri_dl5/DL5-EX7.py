
import matplotlib.pyplot as plt
import numpy as np
import random
import unit10.b_utils as u10

X,Y=u10.load_dataB1W4_trainN()
print(u10.load_dataB1W4_trainN())


def calc_J(X, Y, W, b):
  m = len(Y)
  n = len(W)
  J = 0
  dW = []
  for j in range(n):
    dW.append(0)
  db = 0



  for i in range(m):
    y_hat_i = b
    for j in range(n):
      y_hat_i += W[j]*X[j][i]
    diff = (float)(y_hat_i - Y[i])
    J += (diff**2)/m
    for j in range(n):
      dW[j] += (2*diff/m)*X[j][i]
    db += 2 * (diff);
  return J, dW, db/m

def train_n_adaptive(x,y,alpha,epoch,func):
    n=len(x)
    m=len(y)
    b=0
    W=np.zeros(n)
    
    Aarr=np.repeat(alpha,W.shape[0]+1)
    for i in range(epoch):
        
      
        for i in range(W.shape[0]):
            if dw[i]*Aarr[i]>0:
                Aarr[i]*=1.1
                W[i]-=Aarr[i]
            else:
                    Aarr[i]*=-0.5
                    W[i]-=Aarr[i]

        if db*Aarr[-1]>0:
            Aarr[-1]*1.1
            b-=Aarr[-1]
        else:
            Aarr[-1]*0.5
            b-=Aarr[-1]
    print(i,cost)


    return cost, dw, db
costs, W, b = train_n_adaptive(X, Y, 0.01, 150000, calc_J)
print("costs ="+str(costs)+'W1='+str(W[0])+', W2='+str(W[1])+', W3='+str(W[2])+', W4='+str(W[3])+", b="+str(b))
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per 10,000)')
plt.show()
