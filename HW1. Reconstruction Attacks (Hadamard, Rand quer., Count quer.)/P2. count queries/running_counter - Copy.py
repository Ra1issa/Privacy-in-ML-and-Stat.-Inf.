import random
import numpy as np
import math
import sys
import gc
import helper_functions as hlp
import matplotlib.pyplot as plt
import scipy.linalg



def cases_rand(value, x):
    return {
        'base0': lambda x: [0,1], # if a[0] = 0, then we know that z[0] = 0
        'base1': lambda x: [random.randint(0,1), 0],# if a[0] = 1, then we aren't sure about z and flip a coin
        'base2': lambda x: [1,1], # if a[0] = 2, then we know that z[0] = 1
        '-1': lambda x: [[0,1], [1,1]],
         '0': lambda x: [[0,0], x],
         '1': lambda x: [[0,0], x],
         '2': lambda x: [[1,1] ,[0,1]],
         'backtrack11': lambda x: [1,0],
         'backtrack10': lambda x: x,
         'backtrack01': lambda x: x,
         'backtrack00': lambda x: [0,0],
    }.get(value)(x)

def cases_apriori(value, x):
    return {
        'base0': lambda x: [0,1], # if a[0] = 0, then we know that z[0] = 0
        'base1': lambda x: [random.randint(0,1), 0],# if a[0] = 1, then we aren't sure about z and flip a coin
        'base2': lambda x: [1,1], # if a[0] = 2, then we know that z[0] = 1
        '-1': lambda x: [[0,1], [1,1]],
         '0': lambda x: [[0,0], x],
         '1': lambda x: [[0,0], x],
         '2': lambda x: [[1,1] ,[0,1]],
         'backtrack11': lambda x: [1,0],
         'backtrack10': lambda x: x,
         'backtrack01': lambda x: x,
         'backtrack00': lambda x: [0,0],
    }.get(value)(x)

def mechanism(x):
    a = [sum(x[:i+1])+random.randint(0,1) for i in range(0,len(x))]
    return a

def attacker_random(a):
    lowtri = [[1]*(i+1) + [0]*(len(a) - i - 1) for i in xrange(len(a))]
    x = [0] * len(a)
    z = [[0,0]] * len(a) #z[i][0]: value of z[i], z[i][1]: certainty of z[i]

    # if(a[0] == 2):
    #     z[0] = cases_rand('base2', x)
    # elif(a[0] == 0):
    #     z[0] = cases_rand('base0', x)
    # else:
    #     z[0] = cases_rand('base1', x)

    for i in range(0,len(a)-1) :
        if(i == 0):
            print 'base case'
            diff1 = a[0]
            z[i] = cases_rand('base'+str(diff1), z[i])
        else:
            print 'finding certain z, i:'+ str(i)
            diff1 = a[i] - a[i-1]
            [z[i], z[i-1]]= cases_rand(str(diff1), z[i-1])

    for i in range(1,len(a)-1) :
         diff1 = a[i] - a[i-1]
         if(z[i][1] == 0):
             print 'backtracking, at iteration'+ str(i)
             z[i] = cases_rand('backtrack'+str(diff1)+str(z[i-1][0]), z[i])
         # if(z[i][1] == 0):
         #     if(diff1 == 1):
         #         if(z[i-1]==1):
         #             z[i] = [1,0]
         #     elif(diff1 == 0):
         #          if(z[i-1]==0):
         #              z[i] = [0,0]
    x = np.linalg.solve(np.matrix(lowtri), np.subtract(np.array(a),np.array(hlp.firstelements(z))))
    return x


def attacker_apriori(a, w):
    x = [0] * len(a)
    z = np.random.choice(2, len(a))
    diff = [0] * len(a)
    sure = [0] * len(a)
    lowtri = [[1]*(i+1) + [0]*(len(a) - i - 1) for i in xrange(len(a))]
    # a[i] - a[i-1] =  Z_{i} + x_{i} - Z_{i-1}
    if(a[0] == 2):
        z[0] = 1
        sure[0] = 1
    elif(a[0] == 0):
        z[0] = 0
        sure[0] = 1
    else:
        z[0] = hlp.coin(w[0],-1*w[0]+1,1.0/3.0)
    # First set the Z's that are certain
    for i in range(1,len(a)-1) :
        diff1 = a[i] - a[i-1] # x_i+Z_i-Z_{i-1}
        if(diff1 == 2):
            z[i] = 1
            z[i-1] = 0
            sure[i] = 1
            sure[i-1] = 1
        else: #state 0: 101 011 000 state 1: 111 010 100
            z[i] = hlp.coin(w[i],-1*w[i]+1,1.0/3.0)

    for i in range(1,len(a)-1) :
         diff1 = a[i] - a[i-1]
         if(sure[i] == 0):
             if(diff1 == 1):
                 if(z[i-1]==1):
                     z[i] = 1
    x = np.linalg.solve(np.matrix(lowtri), np.subtract(np.array(a),np.array(z)))
    return x



def enviornment():
    n = [100, 500, 1000, 5000]
    pr = 2.0/3.0
    res1 = []
    res2 = []
    res_rand = []
    res_apriori = []
    for i in range(len(n)):

        for j in range(20):
            x = np.random.randint(2, size= n[i])
            w = [ hlp.coin(x[k],-1*x[k]+1, 2.0/3.0) for k in range(n[i])]
            a = mechanism(x)
            x_r = attacker_random(a)
            #x_i = attacker_apriori(a, w)
            res1.append(hlp.hamming(x,x_r)/float(n[i]))
            #res2.append(hamming(x,x_i)/float(n[i]))
        res_rand.append( np.mean(res1))
        #res_apriori.append( np.mean(res2))

    print list(res_rand)
    print list(n)
    #hlp.formatplot_countqueries(n, res_rand, res_apriori)
    #print_metrics(res_rand, res_apriori)
    plt.show()
    return

enviornment()
