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
    z = np.random.choice(2, len(a))
    diff = [0] * len(a)
    sure = 0

    lowtri = [[1]*(i+1) + [0]*(len(a) - i - 1) for i in xrange(len(a))]
    # a[i+1] - a[i] =  Z_{i+1} + x_{i+1} - Z_i
    # a[i] - a[i-1] =  Z_{i} + x_{i} - Z_{i-1}
    if(a[0] == 2):
        z[0] = 1
    elif(a[0] == 0):
        z[0] = 0
    else:
        z[0] = random.randint(0,1)
    flag = 0

    # First set the Z's that are certain
    for i in range(1,len(a)-1) :
        diff1 = a[i] - a[i-1]
        diff2 = a[i+1] - a[i]
        if(diff1 == 2):
            z[i] = 1
            z[i-1] = 0
        elif(diff2 == -1):
            z[i] = 1
            z[i+1] = 0
        else:
            z[i] = 0
            # if(diff1==0 and diff2==0):
            #     z[i] = coin(1,0,1.0/3.0)
            # elif(diff1==1 and diff2==0):
            #     z[i] = coin(1,0,4.0/5.0)
            # elif(diff1==0 and diff2==1):
            #     z[i] = coin(1,0,1.0/5.0)
            # elif(diff1==1 and diff2==1):
            #     z[i] = coin(1,0,0.5)

    print sure
    print len(a)-sure-1
    x = np.linalg.solve(np.matrix(lowtri), np.subtract(np.array(a),np.array(z)))
    return x


def attacker_inference(a, w):
    x = [0] * len(a)
    z = np.random.choice(2, len(a))
    diff = [0] * len(a)
    sure = 0

    lowtri = [[1]*(i+1) + [0]*(len(a) - i - 1) for i in xrange(len(a))]
    # a[i+1] - a[i] =  Z_{i+1} + x_{i+1} - Z_i
    # a[i] - a[i-1] =  Z_{i} + x_{i} - Z_{i-1}
    if(a[0] == 2):
        z[0] = 1
    elif(a[0] == 0):
        z[0] = 0
    else:
        z[0] = coin(w[0],-1*w[0]+1,1.0/3.0)
    # First set the Z's that are certain
    for i in range(1,len(a)-1) :
        diff1 = a[i] - a[i-1] # x_i+Z_i-Z_{i-1}
        diff2 = a[i+1] - a[i] # x_{i+1}+Z_{i+1}-Z_{i}
        if(diff1 == 2):
            z[i] = 1
            z[i-1] = 0
        elif(diff2 == -1):
            z[i] = 1
            z[i+1] = 0
        else: #state 0: 101 011 000 state 1: 111 010 100
            z[i] = coin(w[i],-1*w[i]+1,1.0/3.0)
            if(diff1==0 and diff2==0): # 000->000, 101->000, 011->101
                prediction = coin(w[i],-1*w[i]+1,1.0/3.0)
                z[i] = coin(prediction,-1*prediction+1,2.0/3.0)
            elif(diff1==1 and diff2==0): # 111->101, 111->011, 010->101, 010->011, 100->000
                prediction = coin(w[i],-1*w[i]+1,2.0/5.0)
                z[i] = coin(prediction,-1*prediction+1,2.0/3.0)
            elif(diff1==0 and diff2==1): # 000->010, 000->100, 101->100, 101->010, 011->111
                prediction = coin(w[i],-1*w[i]+1,2.0/5.0)
                z[i] = coin(prediction,-1*prediction+1,2.0/3.0)
            elif(diff1==1 and diff2==1): # 111->111, 010->111, 100->010, 100->100
                prediction = coin(w[i],-1*w[i]+1,1.0/2.0)
                z[i] = coin(prediction,-1*prediction+1,2.0/3.0)

    print len(a)-sure-1
    x = np.linalg.solve(np.matrix(lowtri), np.subtract(np.array(a),np.array(z)))
    return x



def enviornment():
    n = [100, 500, 1000, 5000]
    pr = 2.0/3.0

    for i in range(len(n)):
        res_rand = []
        res_inf = []
        for j in range(20):
            x = np.random.randint(2, size= n[i])
            w = [ coin(x[k],-1*x[k]+1, 2.0/3.0) for k in range(n[i])]
            a = mechanism(x)
            x_r = attacker_random(a)
            x_i = attacker_inference(a, w)
            res_rand.append(hamming(x,x_r)/float(n[i]))
            res_inf.append(hamming(x,x_i)/float(n[i]))
        rand_av = np.mean(res_rand)
        rand_std = np.std(res_rand)
        print "Random attack average:"+str(rand_av)+" standard deviation:"+str(rand_std)
        inf_av = np.mean(res_inf)
        inf_Std = np.std(res_inf)
        print "Attack with a-priori knowledge average:"+str(inf_av)+" standard deviation:"+str(inf_Std)

    return

enviornment()
