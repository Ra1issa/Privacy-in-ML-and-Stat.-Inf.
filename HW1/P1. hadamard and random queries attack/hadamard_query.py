## Privacy in Machine Learning and Statistical Inference
## Reconstruction Attack:
##The Hadamard Attack and Random Queries
## author: Rawane Issa

import random
import numpy as np
import math
import sys
import helper_functions as hlp
import matplotlib.pyplot as plt
import scipy.linalg


def mechanism(x, H, n , sigma):
    Y = np.random.normal(0, sigma, n)
    a = (1.0/n) * H.dot(x) + Y
    return a

def attacker(a, H, n , sigma):
    z = np.matmul(H,a)
    x = [(0 if a < 0.5 else 1) for a in z]
    return x


def enviornment():
    n = [128,512,2048,8192]
    H = map(hlp.generate_H, n)
    sigma = map(hlp.generate_sigma, n)

    res = []
    for i in range(0,len(n)):

        tmp2 = []
        for k in range(0,len(sigma[i])):
            
            tmp = []
            for j in range(0,20):
                x = np.random.randint(2, size=n[i])
                a = mechanism(x, H[i], n[i], sigma[i][k])
                x2 = attacker(a, H[i], n[i], sigma[i][k])

                tmp.append(hlp.hamming(x,x2)/float(n[i]))
            tmp2.append(np.average(tmp))
        res.append(tmp2)
        bound = np.multiply(np.square(sigma[i]),(n[i]*4))
        hlp.formatplot_hadamard(n,sigma,res,i,bound)

    plt.show()

enviornment()
