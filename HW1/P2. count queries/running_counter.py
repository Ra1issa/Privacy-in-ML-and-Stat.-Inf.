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
def mechanism(x):
    a = [sum(x[:i+1])+random.randint(0,1) for i in range(0,len(x))]
    return a

def attacker_random(a):
    x = [0] * len(a)
    z = [0] * len(a)
    diff = [0] * len(a)
    sure = [0] * len(a)

    lowtri = [[1]*(i+1) + [0]*(len(a) - i - 1) for i in xrange(len(a))]
    # a[i+1] - a[i] =  Z_{i+1} + x_{i+1} - Z_i
    # a[i] - a[i-1] =  Z_{i} + x_{i} - Z_{i-1}
    if(a[0] == 2):
        z[0] = 1
        sure[0]=1
    elif(a[0] == 0):
        z[0] = 0
        sure[0]=1
    else:
        z[0] = random.randint(0,1)
        sure[0]=0
    flag = 0

    # First set the Z's that are certain
    for i in range(1,len(a)-1) :
        diff1 = a[i] - a[i-1]
        if(diff1 == 2):
            z[i] = 1
            z[i-1] = 0
            sure[i]=1
            sure[i-1]=1
        elif(diff1 == -1):
            z[i] = 0
            z[i-1] = 1
            sure[i]=1
            sure[i-1]=1
        else:
            z[i] = 0

    for i in range(1,len(a)-1) :
         diff1 = a[i] - a[i-1]
         if(sure[i] == 0):
             if(diff1 == 1):
                 if(z[i-1]==1):
                     z[i] = 1
             elif(diff1 == 0):
                  if(z[i-1]==0):
                      z[i] = 0
    x = np.linalg.solve(np.matrix(lowtri), np.subtract(np.array(a),np.array(z)))
    return x


def attacker_inference(a, w):
    x = [0] * len(a)
    z = np.random.choice(2, len(a))
    diff = [0] * len(a)
    sure = [0] * len(a)
    lowtri = [[1]*(i+1) + [0]*(len(a) - i - 1) for i in xrange(len(a))]
    # a[i+1] - a[i] =  Z_{i+1} + x_{i+1} - Z_i
    # a[i] - a[i-1] =  Z_{i} + x_{i} - Z_{i-1}
    if(a[0] == 2):
        z[0] = 1
        sure[0] = 1
    elif(a[0] == 0):
        z[0] = 0
        sure[0] = 1
    else:
        z[0] = coin(w[0],-1*w[0]+1,1.0/3.0)
    # First set the Z's that are certain
    for i in range(1,len(a)-1) :
        diff1 = a[i] - a[i-1] # x_i+Z_i-Z_{i-1}
        if(diff1 == 2):
            z[i] = 1
            z[i-1] = 0
            sure[i] = 1
            sure[i-1] = 1
        else: #state 0: 101 011 000 state 1: 111 010 100
            z[i] = coin(w[i],-1*w[i]+1,1.0/3.0)

    for i in range(1,len(a)-1) :
         diff1 = a[i] - a[i-1]
         if(sure[i] == 0):
             if(diff1 == 1):
                 if(z[i-1]==1):
                     z[i] = 1
             # elif(diff1 == 0):
             #      if(z[i-1]==0):
             #          z[i] = 0
    x = np.linalg.solve(np.matrix(lowtri), np.subtract(np.array(a),np.array(z)))
    return x



def enviornment():
    n = [100, 500, 1000, 5000]
    pr = 2.0/3.0
    res1 = []
    res2 = []
    res_rand = []
    res_inf = []
    for i in range(len(n)):

        for j in range(20):
            x = np.random.randint(2, size= n[i])
            w = [ coin(x[k],-1*x[k]+1, 2.0/3.0) for k in range(n[i])]
            a = mechanism(x)
            x_r = attacker_random(a)
            x_i = attacker_inference(a, w)
            res1.append(hamming(x,x_r)/float(n[i]))
            res2.append(hamming(x,x_i)/float(n[i]))
        res_rand.append( np.mean(res1))
        res_inf.append( np.mean(res2))

    print list(res_rand)
    print list(n)
    plt.plot(list(n), list(res_rand),'-o', label='rand')
    plt.plot(list(n), list(res_inf),'-o', label='apriori')
    #plt.plot(list(sigma[i]), list(bound),'-', label=' bound')
    plt.ylabel('Fraction. Error (Ham/n)')
    plt.xlabel('n')
    #plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,wspace=0.35)
    plt.legend(loc='best')

    rand_av = np.mean(res_rand)
    rand_std = np.std(res_rand)
    print "Random attack average:"+str(rand_av)+" standard deviation:"+str(rand_std)
    inf_av = np.mean(res_inf)
    inf_Std = np.std(res_inf)
    print "Attack with a-priori knowledge average:"+str(inf_av)+" standard deviation:"+str(inf_Std)
    plt.show()
    return

enviornment()
