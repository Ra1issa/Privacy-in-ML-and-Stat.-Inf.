## Privacy in Machine Learning and Statistical Inference
## Count Queries:
## author: Rawane Issa
import random
import numpy as np
import math
import sys
import gc
import matplotlib.pyplot as plt
import scipy.linalg

hamming = lambda n, m: sum([ int(n[i]) ^ int(m[i]) for i in range(len(n))   ]  )
myround = lambda x: 1 if x >= 0.5 else 0
coin  = lambda x,y,p: x if(random.random() < p) else y
firstelements = lambda x: [x[i][0] for i in range(len(x))]

def formatplot_countqueries(n, res_rand, res_apriori):
    name = 'plots/count/countqueries.png'

    plt.plot(list(n), list(res_rand),'-o', label='rand')
    plt.plot(list(n), list(res_apriori),'-o', label='apriori')

    plt.ylabel('Fraction. Error (Ham/n)')
    plt.xlabel('n')
    plt.legend(loc='best')
    
    plt.savefig(name)


def print_metrics(res_rand, res_apriori):
    rand_av = np.mean(res_rand)
    rand_std = np.std(res_rand)
    print "Random attack average:"+str(rand_av)+" standard deviation:"+str(rand_std)

    apriori_av = np.mean(res_apriori)
    apriori_Std = np.std(res_apriori)
    print "Attack with a-priori knowledge average:"+str(apriori_av)+" standard deviation:"+str(apriori_Std)
