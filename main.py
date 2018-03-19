#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 14:12:22 2018

@author: alexandreboyker
"""

import numpy as np
from manifold_regularizer import ManifoldRegularizer
import matplotlib.pyplot as plt
from helper import get_2_moons, get_isolet
from matplotlib  import cm
from sklearn.model_selection import train_test_split
import os

data_path = 'data'

kernel = lambda x, y: np.exp(-np.linalg.norm(x-y)**2 )
#kernel = lambda x,y: np.dot(x,y)

gamma_i = 10.0
gamma_a = 24.03125

def main():
    
    X, y= get_2_moons()
    plt.scatter(X[:,0], X[:,1], marker='X',c=np.append(y, 3*np.ones(X[:,0].size-y.size)), cmap=cm.jet)
    plt.show()
    plt.scatter(X[:y.size,0], X[:y.size,1], marker='X',c=y, cmap=cm.jet)
    plt.show()
    print(y)
    param = {'kernel':kernel, 'gamma_i':gamma_i, 'gamma_a':gamma_a}
    mr = ManifoldRegularizer(**param)
    mr.fit(X, y)
    
    isolet_data = get_isolet(os.path.join(data_path, 'isolet.csv'))
    
    
    
if __name__ == '__main__':
    
    main()