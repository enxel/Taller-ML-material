import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()

print(iris.DESCR)

print(iris.keys())

print(iris.target_names)

n_samples, n_features = iris.data.shape
print('Number of samples:', n_samples)
print('Number of features:', n_features)

print(iris.data)

print(iris.data[0])

print(iris.data[[12, 26, 89, 114]])

print(iris.data.shape)
print(iris.target.shape)
print(iris.target)

print(np.bincount(iris.target))

print(iris.data[iris.target==1])
print(iris.data[iris.target==1][:5])
print(iris.data[iris.target==1, 0][:5])
