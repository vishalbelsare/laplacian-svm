#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 17:44:51 2018

@author: alexandreboyker
"""

from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

# Loading some example data
iris = datasets.load_iris()
X = iris.data[:, [0, 2]]
y = iris.target
# Training classifiers

clf3 = SVC(kernel='rbf', probability=True)

clf3.fit(X, y)

# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
print(xx.shape, yy.shape)
f, axarr = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(10, 8))
for idx, clf, tt in zip(product([0, 1], [0, 1]), [clf3], ['Kernel SVM']):
    print(np.c_[xx.ravel(), yy.ravel()].shape)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    print(Z.shape)
    Z = Z.reshape(xx.shape)
    axarr.contourf(xx, yy, Z, alpha=0.4)
    axarr.scatter(X[:, 0], X[:, 1], c=y,
                                  s=20, edgecolor='k')
    axarr.set_title(tt)
    

plt.show()






from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import numpy as np


digits = load_digits()
X, y = shuffle(digits.data, digits.target)
X_train, X_test = X[:1000, :], X[100:, :]
y_train, y_test = y[:1000], y[100:]

svc = SVC(kernel='precomputed')

kernel_train = np.dot(X_train, X_train.T)  # linear kernel
print(kernel_train.shape, X_train.shape)
svc.fit(kernel_train, y_train)

#kernel_test = np.dot(X_test, X_train[svc.support_, :].T)
kernel_test = np.dot(X_test, X_train.T)
print(kernel_test.shape, X_test.shape)

y_pred = svc.predict(kernel_test)
print ('accuracy score: %0.3f' % accuracy_score(y_test, y_pred))