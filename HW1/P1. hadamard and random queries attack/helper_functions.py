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

hamming = lambda n, m: sum([ int(n[i]) ^ int(m[i]) for i in range(len(n))   ]  )
# myround = lambda x: 1 if x >= 0.5 else 0

generate_m = lambda x: [int(round(1.1*x)), 4*x, 16*x]
generate_sigma = lambda x: [2**-j for j in range(1,np.log2(math.sqrt(32*x)).astype(int) + 1)]
generate_H = lambda x: scipy.linalg.hadamard(x)

def  format_subplot(n,sigma,res,i,m):
    #plt.subplot(2,2,i+1)
    title = 'n:'+str(n[i])
    plt.title(title)
    plt.plot(list(sigma[i]), list(res[0]),'-o', label=' m:'+str(m))
    plt.xlim(0,0.5)
    plt.ylim(0,60)
    plt.ylabel('Perc. Error (100*Ham/n)')
    plt.xlabel('sigma')
    plt.legend(loc='best')
    name = 'n'+str(n[i])+'m'+str(m)+'.png'
    plt.savefig(name)

def format_plot():
    plt.ylabel('Perc. Error (100*Ham/n)')
    plt.xlabel('sigma')
    #plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,wspace=0.35)
    plt.legend(loc='best')
    plt.show()
