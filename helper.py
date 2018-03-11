#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 16:58:54 2018

@author: alexandreboyker
"""


from sklearn.datasets import make_moons


def get_2_moons():
    
    return make_moons(n_samples=200, shuffle=True, noise=.05, random_state=None)


