#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 15:11:37 2018

@author: alexandreboyker
"""
from adj_matrix import AdjacencyMatrix
import numpy as np
from sklearn import svm

import matplotlib.pyplot as plt
from itertools import product


class ManifoldRegularizer(object):
    
    def __init__(self, kernel=None, gamma_i=1, gamma_a=0.03125):
        
        if kernel is None:
            self.kernel = np.dot
            
        else: 
            
            self.kernel = kernel
        
        
        self.threshold = self.kernel(np.array([0,0,0]), np.array([0,0,0]))
        self.gamma_a = gamma_a
        
        self.gamma_i = gamma_i
        
        self.n_labels = 0
    
    def _create_diagonal_matrix(self, array):
        
        size = array.size
        
        mat = np.zeros((size, size))
        
        for i in range(size):
            
            mat[i][i] = array[i]
            
        return mat
    
    def _get_Q(self, K, L, X):
        
        self.mapping_beta_to_alpha = np.linalg.inv(2 * self.gamma_a * np.identity(K.shape[0]) +2 * self.gamma_i * (1/(K.shape[0])**2 * np.dot(L,K)))
        return self.mapping_beta_to_alpha

    def get_alphas(self, clf, X, y):
        
        beta = np.zeros(y.size)
        beta[clf.support_] = clf.dual_coef_

        J = np.concatenate((np.identity(y.size), np.zeros((y.size, X.shape[0]-y.size))), axis=1)
        y = self._create_diagonal_matrix(y)
        return np.dot(np.dot(np.dot(self.mapping_beta_to_alpha, J.T),y), beta)
        
    
    def _get_W_K_L(self, X):
        
        mat = AdjacencyMatrix(distance=self.kernel)
        W = mat.fit(X)
        
        mat = AdjacencyMatrix(distance=self.kernel)
        K = mat.fit(X)
        
        array_distance = np.array([ np.sum(W[i]) for i in range(X.shape[0]) ])
        
        L = self._create_diagonal_matrix(array_distance) - W
        
        
        return W, K, L
    
    def _get_regularized_kernel(self, X):
        
        W, K, L = self._get_W_K_L(X)
        return np.dot(K, self._get_Q(K, L, X))
    
    def _get_adjusted_regularized_kernel(self, X, y):
                
        J = np.concatenate((np.identity(y.size), np.zeros((y.size, X.shape[0]-y.size))), axis=1)
        reg_ker = self._get_regularized_kernel(X)
        return np.dot(np.dot(J,reg_ker) , J.T)

    def plot_decision_region(self, X, y, svm_classifier):
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        
        f, axarr = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(10, 8))
        
        for idx, clf, tt in zip(product([0, 1], [0, 1]),
                                [svm_classifier],
                                ['Kernel SVM']):
        
            Z= np.c_[xx.ravel(), yy.ravel()] 
            
            Z = self.predict(Z)
    
            Z = Z.reshape(xx.shape)
            axarr.contourf(xx, yy, Z, alpha=0.4)
            axarr.scatter(X[:, 0], X[:, 1], c=y,
                                          s=20, edgecolor='k')
            axarr.set_title(tt)
            
        
        plt.show()
    

    def predict(self, X_test):
        predi = []
        decision = lambda x: 1 if x>=self.threshold else 0
        for j in range(X_test.shape[0]):
            
            predi.append( sum([self.alpha[i] * self.kernel(self.X[i], X_test[j]) for i in range(self.X.shape[0])]))
        predi = [decision(i) for i in predi]
        return np.array(predi)
    
    def fit(self, X, y):
        self.X = X
        self.size = X.shape[0]
        self.n_labels = y.size
        self.n_unlabelled_points = self.size - self.n_labels 
        
        regularized_kernel = self._get_adjusted_regularized_kernel(X, y)
        

        clf = svm.SVC(kernel='precomputed', C=self.gamma_a)
        clf.fit(regularized_kernel, y) 
        self.alpha = self.get_alphas(clf, X, y)
        
        self.plot_decision_region(X, np.append(y, 3*np.ones(X[:,0].size-y.size)), clf)