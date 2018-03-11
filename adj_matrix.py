#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 14:15:25 2018

@author: alexandreboyker
"""
import numpy as np

import time
from scipy.spatial.distance import pdist


class AdjacencyMatrix(object):
    
    def __init__(self, distance=None):
        
        if distance is None:
            
            self.distance = lambda x,y: np.linalg.norm(x-y)

        else:
            
            #self.distance = lambda x, y: np.exp(-np.linalg.norm(x-y)**2 * (1/4))
            self.distance = distance
            
    def fit(self, X):
        
        if type(X) is not np.ndarray:
            
            raise ValueError('X should be numpy one-dimensional array')


        start = time.time()
        
        size = X.shape[0]
        #distance_matrix = pdist(X, self.distance)
        distance_matrix = np.zeros((size, size))
        
        for i in range(size):
            
            for j in range(i+1):
                
                distance_matrix[i][j] = self.distance(X[i], X[j])
                
                distance_matrix[j][i] = distance_matrix[i][j]
        
        
        
        
        
        end = time.time()
        #print('time taken by {} method {}: {} seconds'.format(self.__str__(), self.fit.__name__, end - start))
        
        
        return distance_matrix

        
        
        