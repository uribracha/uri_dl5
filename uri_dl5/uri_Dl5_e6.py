import unit10.b_utils as u10
import matplotlib.pyplot as plt
import random 


def calc_J(X, Y, a, b):
    m = len(Y)
    sumJ = 0
    sumDa = 0
    sumDb = 0
    for i in range (m):
        sumJ += Li(a,b,X[i], Y[i])
        sumDa += 2*X[i]*(a*X[i]+b-Y[i])
        sumDb += 2*(a*X[i]+b-Y[i])
    return sumJ/m, sumDa/m, sumDb/m

def train_adaptive(x,y,alpha, epocs, MyFunc):
    a,b=(random.randrange(-10,10),random.randrange(-10,10))
    a_Alpha=alpha
    B_alpha=alpha
    for i in range (epocs):
        sumj, sumda,sumdb = MyFunc(x,y,a,b)
        if (sumj*a_Alpha<0): 
           a_Alpha*= 1.1   
        else:
            a_Alpha *= -0.5 

        if (sumj*B_alpha<0): 
            B_alpha*= 1.1   
        else:
            B_alpha *= -0.5 

        a +=np.abs(sumj)*a_Alpha
        b+=np.abs(sumj)*B_alpha
        Fx, Fdx = MyFunc(x,y,a,b)
    return  Fx,a,b

X, Y = u10.load_dataB1W3Ex1()
J, a, b = train_adaptive(X, Y, 0.001, 1000, calc_J)
print('J='+str(J)+', a='+str(a)+", b="+str(b))
plt.plot(X, Y, 'r.')
plt.plot([0,100],[a*0+b, 100*a+b])
plt.show()