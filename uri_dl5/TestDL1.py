import numpy as np

import matplotlib.pyplot as plt

import sklearn

import sklearn.datasets

import sklearn.linear_model

from unit10 import c1w3_utils as u10

from DL1 import *
np.random.seed(1)

X, Y = u10.load_planar_dataset()

plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral);

plt.show()
d=u10.load_planar_dataset()
clf = sklearn.linear_model.LogisticRegressionCV();

clf.fit(X.T, Y[0,:])

# Plot the decision boundary for logistic regression

u10.plot_decision_boundary(lambda x: clf.predict(x), X, Y)

plt.title("Logistic Regression")

plt.show()

# Print accuracy

LR_predictions = clf.predict(X.T)

print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions)
+ np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +

'% ' + "(percentage of correctly labelled datapoints)")

np.random.seed(1)
Model=DLModel("model")

layer1=DLLayer("LAYER1",70,(2,),"leaky_relu",learning_rate=0.3,optimization="adaptive")
layer2=DLLayer("LAYER2",70,(70,),"leaky_relu",learning_rate=0.3,optimization="adaptive")
layer3=DLLayer("LAYER3",70,(70,),"leaky_relu",learning_rate=0.3,optimization="adaptive")
layer4=DLLayer("LAYER4",30,(70,),"leaky_relu",learning_rate=0.3,optimization="adaptive")

layer5=DLLayer("LAYER5",1,(70,),"sigmoid",learning_rate=0.3,optimization="adaptive")
Model.compile("cross_entropy")
Model.add(layer1)
Model.add(layer2)
Model.add(layer3)
Model.add(layer5)
print(Model)

costs = Model.train(X,Y,int(1e5))

plt.plot(np.squeeze(costs))

plt.ylabel('cost')

plt.show()

u10.plot_decision_boundary(lambda x: Model.predict(x.T), X, Y)

plt.title("Decision Boundary for hidden layer size " + str(4))

plt.show()

predictions = Model.predict(X)

print ('Accuracy: %d' % float((np.dot(Y,predictions.T) +
np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')


X, Y = u10.load_planar_dataset()

noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure =u10.load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,

"noisy_moons": noisy_moons,

"blobs": blobs,

"gaussian_quantiles": gaussian_quantiles}

dataset = "noisy_moons"

X, Y = datasets[dataset]

X, Y = X.T, Y.reshape(1, Y.shape[0])

if dataset == "blobs":
    Y = Y%2
