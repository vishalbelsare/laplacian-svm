#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 16:58:54 2018

@author: alexandreboyker
"""


from sklearn.datasets import make_moons
import pandas as pd
import numpy as np
def get_2_moons():
    
    X, y = make_moons(n_samples=200, shuffle=True, noise=.03, random_state=None)
    n = 6
    x_0 = X[y==0][:n]
    y_0 = [0 for item in range(x_0.shape[0])]

    x_1 = X[y==1][:n]

    y_1 = [1 for item in range(x_1.shape[0])]
    
    X=  np.concatenate((np.concatenate([x_0, x_1], axis=0), np.concatenate([X[y==0][n:], X[y==1][n:]], axis=0)), axis=0)
    return X, np.array(y_0 + y_1)


def get_isolet(path):
    
    data = pd.read_csv(path)
    data['class'] = data['class'].apply(lambda x: int(''.join(e for e in x if e.isalnum())))
    return data