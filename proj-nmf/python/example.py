#!/usr/bin/env python
import numpy as np
from numpy import *
from nmf import *

#w1 = np.array([[1,2,3],[4,5,6]])
#h1 = np.array([[1,2],[3,4],[5,6]])
#w2 = np.array([[1,1,3],[4,5,6]])
#h2 = np.array([[1,1],[3,4],[5,6]])

w1 = np.array([[100,200,300],[400,500,600]])
h1 = np.array([[100,200],[300,400],[500,600]])
w2 = np.array([[100,100,300],[400,500,600]])
h2 = np.array([[100,100],[300,400],[500,600]])

v = np.dot(w1,h1)

(wo,ho) = nmf(v, w2, h2, 0.0001, 100, 1000)
print(wo)
print(ho)
vo = np.dot(wo, ho)
print(v)
print(vo)
