## Privacy in Machine Learning and Statistical Inference
## Reconstruction Attack:
##The Hadamard Attack and Random Queries
## author: Rawane Issa

import random
import numpy as np
import math
import sys
import csv
import gc
import helper_functions as hlp
import matplotlib.pyplot as plt
import scipy.linalg
import numpy.linalg


def mechanism(x, B, m, n, sigma):
    print m, n
    Y = np.random.normal(0, sigma, m)
    print Y.shape
    a = (1.0/n) * B.dot(x)
    a = [ float(a[i] + Y[i]) for i in range(m) ]
    print a[0]
    return a

def attacker(a, B, sigma):
    a = np.matrix(a)
    z = numpy.linalg.lstsq((1.0/B.shape[1])*B, np.transpose(a))[0]
    x = [(0 if a < 0.5 else 1) for a in z]
    return x


def enviornment():

    n = [128,512,2048,8192]
    m = map(hlp.generate_m, n)
    sigma = map(hlp.generate_sigma, n)

    for i in range(2, len(n)):
        plt.clf()
        m = [int(round(1.1*n[0])), 4*n[0], 16*n[0]]
        for l in range(0, len(m)):
            tmp2 = []
            B = np.matrix(np.random.randint(2, size=(m[l], n[i])))

            res = []
            for k in range(0,len(sigma[i])):

                tmp = []
                for j in range(0,20):
                    x = np.random.randint(2, size=(n[i],1))
                    a = mechanism(x, B, m[l], n[i], sigma[i][k])
                    x2 = attacker(a, B, sigma[i][k])
                    tmp.append(hlp.hamming(x,x2)*100/float(n[i]))
                print sum(tmp)/len(tmp)
                tmp2.append(sum(tmp)/len(tmp))

            res.append(tmp2)
            hlp.format_subplot(n,sigma,res,i,m[l])

            bound = np.multiply(np.square(sigma[i]),(n[i]*4))


    #hlp.format_plot()

enviornment()
