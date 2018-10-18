import random
import numpy as np
import math
import sys
import gc
import helper_functions as hlp
import matplotlib.pyplot as plt
import scipy.linalg

# diff=a[i] - a[i-1] = x_i + z_i - z_{i-1}
# Certain vs Non-Certain cases:
# if diff = 2 then (x_i,z_i,z_{i-1}) = (1,1,0) no other possibility
# if diff = -1 then (x_i,z_i,z_{i-1}) = (0,0,1) no other possibility
# if diff = 0 then (x_i,z_i,z_{i-1}) =  (1,0,1) or (0,1,1) or (0,0,0)
# if diff = 1 then (x_i,z_i,z_{i-1}) =  (1,1,1) or (0,1,0) or (1,0,0)

def cases_rand(value, x):
    return {
        'base0': lambda x: [0,1], # if a[0] = 0, then we know that z[0] = 0
        'base1': lambda x: [random.randint(0,1), 0],# if a[0] = 1, then we aren't sure about z and flip a coin
        'base2': lambda x: [1,1], # if a[0] = 2, then we know that z[0] = 1
        '-1': lambda x: [[0,1], [1,1]], #if diff = -1, the with certainty x_i=0,z_i=0,z_{i-1}=1
         '0': lambda x: [[0,0], x], #if diff = 0, we are not sure and just set z_i:=0
         '1': lambda x: [[0,0], x], #if diff= 1, we are not sure and just set z_i:=0
         '2': lambda x: [[1,1] ,[0,1]], #if diff = 2, the with certainty x_i=1,z_i=1,z_{i-1}=0
         'backtrack11': lambda x: [1,0], #if diff = 1 and z_{i-1}=1, then z_i=1 with large likelyhood
         'backtrack10': lambda x: x, #if diff = 1 and z_{i-1}=0, there are multiple ways to pick z_i, do nothing
         'backtrack01': lambda x: x, #if diff = 0 and z_{i-1}=1, there are multiple ways to pick z_i, do nothing
         'backtrack00': lambda x: [0,0],#if diff= 0 and z_{i-1}=0, then z_i=1 with large likelyhood
    }.get(value)(x)


# In the apriori case, we will assume that we have a correct guess of x
# (and be right 2/3 of the time) and do the following for the uncertain case:
#
# if diff = 0 then (x_i,z_i,z_{i-1}) =  (1,0,1) or (0,1,1) or (0,0,0) each with equal likelyhood:
#    in this cases z_i != x_i for (1,0,1) and (0,1,1) i.e. 2/3 of the time
#    so with probability 2/3 we set z_i to !x_i and
#    with probability 1/3 we it to x_i
#
# if diff = 1 then (x_i,z_i,z_{i-1}) =  (1,1,1) or (0,1,0) or (1,0,0):
#    in this cases z_i != x_i for (0,1,0) or (1,0,0) i.e. 2/3 of the time
#    so with probability 2/3 we set z_i to !x_i and (since each event is equally likely)
#    with probability 1/3 we it to x_i


def cases_apriori(value, x):
    return {
        'base0': lambda x,y: [0,1],
        'base1': lambda x,y: [-1*y, 0],# if a[0]=1 then Z[0] = !x[0]
        'base2': lambda x,y: [1,1],
        '-1': lambda x,y: [[0,1], [1,1]],
         '0': lambda x,y: [[hlp.coin(y,-1*y+1,1.0/3.0),0], x],
         '1': lambda x,y: [[hlp.coin(y,-1*y+1,1.0/3.0),0], x],
         '2': lambda x: [[1,1] ,[0,1]],
         'backtrack11': lambda x,y: [1,0],
         'backtrack10': lambda x,y: x,
         'backtrack01': lambda x,y: x,
         'backtrack00': lambda x,y: [0,0],
    }.get(value)(x)

def mechanism(x):
    a = [sum(x[:i+1])+random.randint(0,1) for i in range(0,len(x))]
    return a

def attacker_random(a):
    lowtri = [[1]*(i+1) + [0]*(len(a) - i - 1) for i in xrange(len(a))]
    x = [0] * len(a)
    z = [[0,0]] * len(a)

    # Look for certain cases of Z
    for i in range(0,len(a)-1) :
        if(i == 0):
            print 'base case'
            diff1 = a[0]
            z[i] = cases_rand('base'+str(diff1), z[i])
        else:
            print 'finding certain z, i:'+ str(i)
            diff1 = a[i] - a[i-1]
            [z[i], z[i-1]]= cases_rand(str(diff1), z[i-1])

    # Backtrack to find other possible nearly-certain cases of Z
    for i in range(1,len(a)-1) :
         diff1 = a[i] - a[i-1]
         if(z[i][1] == 0):
             print 'backtracking, at iteration'+ str(i)
             z[i] = cases_rand('backtrack'+str(diff1)+str(z[i-1][0]), z[i])

    values_z = np.array(hlp.firstelements(z))
    x = np.linalg.solve(np.matrix(lowtri), np.subtract(np.array(a),values_z))
    return x


def attacker_apriori(a, w):
    lowtri = [[1]*(i+1) + [0]*(len(a) - i - 1) for i in xrange(len(a))]
    x = [0] * len(a)
    z = [[0,0]] * len(a)

    # Look for certain cases of Z
    for i in range(1,len(a)-1) :
        if(i == 0):
            print 'base case'
            diff1 = a[0]
            z[i] = cases_rand('base'+str(diff1), z[i])
        else:
            print 'finding certain z, i:'+ str(i)
            diff1 = a[i] - a[i-1]
            [z[i], z[i-1]]= cases_rand(str(diff1), z[i-1])

    # Backtrack to find other possible nearly-certain cases of Z
    for i in range(1,len(a)-1) :
        diff1 = a[i] - a[i-1]
        if(z[i][1] == 0):
             print 'backtracking, at iteration'+ str(i)
             z[i] = cases_rand('backtrack'+str(diff1)+str(z[i-1][0]), z[i])

    values_z = np.array(hlp.firstelements(z))
    x = np.linalg.solve(np.matrix(lowtri), np.subtract(np.array(a),values_z))
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
            x_i = attacker_apriori(a, w)

            res1.append(hlp.hamming(x,x_r)/float(n[i]))
            res2.append(hlp.hamming(x,x_i)/float(n[i]))
        res_rand.append( np.mean(res1))
        res_apriori.append( np.mean(res2))

    hlp.formatplot_countqueries(n, res_rand, res_apriori)
    hlp.print_metrics(res_rand, res_apriori)
    plt.show()

enviornment()
