import random
import numpy as np
import math
import sys
import gc
import matplotlib.pyplot as plt
import scipy.linalg


def mechanism(x):
    a = [x[i]+random.choice([0,1]) for i in range(0,len(x))]
    return a


def attacker_random(a):
    return 0

def attacker_inference(a, w):
    return 0

def enviornment():
    n = 5000
    pr = 2.0/3.0

    x = [random.choice([0,1]) for i in range(0,n)]
    w = [np.random.choice([x[i], !x[i]],2,p=[pr, 1-pr])]

    a = mechanism(x)
    x_r = attacker_random(a)
    x_i = attacker_inference(a, w)

    return
