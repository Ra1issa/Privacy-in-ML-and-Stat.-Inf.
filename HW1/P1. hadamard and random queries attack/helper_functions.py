## Privacy in Machine Learning and Statistical Inference
## Reconstruction Attack:
##The Hadamard Attack and Random Queries
## author: Rawane Issa
import random
import numpy as np
import math
import sys
import gc
import matplotlib.pyplot as plt
import scipy.linalg

hamming = lambda n, m: sum([ int(n[i]) ^ int(m[i]) for i in range(len(n))])
generate_m = lambda x: [int(round(1.1*x)), 4*x, 16*x]
generate_sigma = lambda x: [2**-j for j in range(1,np.log2(math.sqrt(32*x)).astype(int) + 1)]
generate_H = lambda x: scipy.linalg.hadamard(x)

def  formatplot_randomattack(n,sigma,res,i,m):
    title = 'Error Growth of rand. reconst.n attack as a function of sigma, for n:'+str(n[i])
    name = 'plots/random/n'+str(n[i])+'m'+str(m)+'.png'

    plt.title(title)
    plt.plot(list(sigma[i]), list(res),'-o', label='value of m='+str(m))

    plt.xlim(0,sigma[i][0])
    plt.ylim(0,0.6)

    plt.ylabel('Fraction. Error (Ham/n)')
    plt.xlabel('sigma')

    plt.legend(loc='best')
    plt.savefig(name)

def formatplot_hadamard(n,sigma,res,i,bound):
    title = 'Error Growth of Hadamard reconstruction attack as a function of sigma and n'
    name = 'plots/hadamard/hadamard_superimposed.png'

    plt.title(title)
    plt.plot(list(sigma[i]), list(res[i]),'-o', label='n:'+str(n[i]))

    plt.xlim(0,0.5)
    plt.ylim(0,0.6)

    plt.ylabel('Fraction. Error (Ham/n)')
    plt.xlabel('sigma')
    
    plt.legend(loc='best')
    plt.savefig(name)
