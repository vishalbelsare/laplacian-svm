#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 14:12:22 2018

@author: alexandreboyker
"""

import numpy as np
from manifold_regularizer import ManifoldRegularizer
import matplotlib.pyplot as plt
from helper import get_2_moons
from matplotlib  import cm
from sklearn.model_selection import train_test_split



kernel = lambda x, y: np.exp(-np.linalg.norm(x-y)**2 *2)
#kernel = lambda x,y: np.dot(x,y)

gamma_i = 1.0
gamma_a = 0.03125

def main():
    
    X, y= get_2_moons()
    X_train, X_test, _, y = train_test_split(X, y, test_size=20, random_state=2)
    X = np.concatenate((X_test, X_train), axis=0)
    
    plt.scatter(X[:,0], X[:,1], marker='o',c=np.append(y, 3*np.ones(X[:,0].size-y.size)), cmap=cm.jet)
    plt.show()
    param = {'kernel':kernel, 'gamma_i':gamma_i, 'gamma_a':gamma_a}
    mr = ManifoldRegularizer(**param)
    mr.fit(X, y)
    
    
    
if __name__ == '__main__':
    
    main()