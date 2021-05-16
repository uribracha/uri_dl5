import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from unit10 import c1w4_utils as u10
from DL1 import *

train_X, train_Y, test_X, test_Y = u10.load_dataset()
np.random.seed(1)
hidden1 = DLLayer("Perseptrons 1", 30,(12288,),"relu",W_initialization = "Xaviar",learning_rate = 0.0075, optimization='adaptive')
hidden2 = DLLayer("Perseptrons 2", 15,(30,),"trim_sigmoid",W_initialization = "He",learning_rate = 0.1)
print(hidden1)
print(hidden2)


hidden1 = DLLayer("Perseptrons 1", 10,(10,),"relu",W_initialization = "Xaviar",learning_rate = 0.0075)

hidden1.b = np.random.rand(hidden1.b.shape[0], hidden1.b.shape[1])

hidden1.save_weights("SaveDir","Hidden1")

hidden2 = DLLayer ("Perseptrons 2", 10,(10,),"trim_sigmoid",W_initialization =
"SaveDir/Hidden1.h5",learning_rate = 0.1)

print(hidden1)

print(hidden2)

model = DLModel()

model.add(hidden1)

model.add(hidden2)

dir = "model"

model.save_weights(dir)

print(os.listdir(dir))



l1=DLLayer("l1",10,(2,),"relu","zeros",0.01)
l2=DLLayer("l2",5,(10,),"relu","zeros",0.01)
l3=DLLayer("l3",1,(5,),"relu","zeros",1.0)
model=DLModel()
model.add(l1)
model.add(l2)
model.add(l3)
model.compile("cross_entropy")
costs = model.train(train_X, train_Y, 15000)
plt.plot(costs)

plt.ylabel('cost')

plt.xlabel('iterations (per 150s)')

axes = plt.gca()

axes.set_ylim([0.65,0.75])

plt.title("Model with -zeros- initialization")

plt.show()
