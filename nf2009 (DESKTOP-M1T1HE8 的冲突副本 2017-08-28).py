# -*- coding: utf-8 -*-
"""
Created on Sat May 14 09:44:03 2016

@author: Evojoy
"""
import numpy as np
import toolbox as ft
import scipy as sy
from scipy.integrate import quad
from math import pi, exp
import pandas as pd
#df = np.log( pd.read_excel("399300.xlsx", "399300", index_col = 0) + 1 ).resample("Q").sum() # this line gets the main economic inputs
corr, std_2, std_3 =  0.160755, 0.223732, 0.345276
#corr, std_2, std_3 = 0.1, 0.1, 0.1
size = 10000 * 30

Rf = 1 + 0.02852 # inputs
#Rf = 1 + 0.02 # inputs
g2,g3 = 0.087346, 0.083169 # real estate, stocks
#g2,g3 = 0.04, 0.04 # inputs

covs = [[1,    corr], 
       [corr,  1]] 
m = np.random.multivariate_normal([0,0], covs, size)
x = m[:,0]
y = m[:,1]
RR2 = np.exp(g2 + std_2 * x) # housing
RR3 = np.exp(g3 + std_3 * y) # stock
RRf = np.ones(size)* Rf
    
def optimal_w3(Gamma, Beta, loss_aversion, B0, gm, delta, first_guess, W2):
    gamma, beta, LA, b0, gm, delta = Gamma, Beta, loss_aversion, B0, gm, delta
    w2 = W2
    w3_lower_bound = 0.0001
    w3_upper_bound= 1-w2
    # get proper numbers of testing w3, whatever w2 is
    digs = round((w3_upper_bound-w3_lower_bound)*100+1,2)
    
    def B(w3, Alpha):
        alpha = Alpha
        port = (RRf*(1-w2-w3) + RR2 * w2 + RR3 * w3) # portfolio wealth
        Exp = port **(1-gamma)
        E = (Exp.mean())**(1/(1-gamma)) 
        first = (1-beta)**(1/(1-gamma))*alpha**(-gamma/(1-gamma)) * E
        # here we also use a numerical method to estimate the expectation of gains and losses
        log_excess = (RR3-RRf)* w3
        pos = ((log_excess[log_excess>=0])**gm)**delta
        POS = pos.mean()*(len(pos)/float(size))
        neg = ((-log_excess[log_excess<0])**gm)**delta
        NEG = -neg.mean()*(len(neg)/float(size))
        #print pos, neg
        second = b0 * (POS + LA * NEG)
        return first + second
    def new_alpha(alpha):
        result = [] 
        for w3 in np.linspace(w3_lower_bound,w3_upper_bound,digs):
            result.append(B(w3, alpha))    
        df = pd.DataFrame(result, index = np.linspace(w3_lower_bound,w3_upper_bound,digs))
        ans = df.max().values[0]
        #print "current w3", df.idxmax()[0]
        def f(a):
            y = (1-beta)*a**(-gamma)- beta*(1-a)**(-gamma) * (ans**(1-gamma))
            return y   
        new_a = sy.optimize.fsolve(f, 0.01)[0]
        return new_a
        
    def find_alpha(guess):
        initial_guess = guess
        last_alpha = float(new_alpha(initial_guess))
        next_alpha = new_alpha(last_alpha)
        diff = next_alpha - last_alpha
        while (diff<-0.01 or diff>0.01):
            last_alpha = float(next_alpha)
            next_alpha = new_alpha(last_alpha)
            diff = next_alpha - last_alpha
            #print next_alpha
        #print "alpha", next_alpha
        return next_alpha
        
    def W3(alpha):
        result = [] 
        for w3 in np.linspace(w3_lower_bound,w3_upper_bound,digs):
            result.append(B(w3, alpha))    
        df = pd.DataFrame(result, index = np.linspace(w3_lower_bound,w3_upper_bound,digs))
        #ans = df.idxmax()[0]
        return df
    df = W3(float(find_alpha(first_guess)))
    return round(df.idxmax()[0],3)

w_list= []

begin, start, num = 0, .15, 16

for b in np.linspace(begin, start, num):
    print b
    w_list.append(optimal_w3(2.5, 0.98, 2.25, b, .88, 1, 0.025, 0.6))


df = pd.DataFrame(data = np.linspace(begin, start, num), index = w_list, columns = ["b0"] )
print df

w_list= []
for b in np.linspace(begin, start, num):
    print b
    w_list.append(optimal_w3(4.0, 0.98, 2.25, b, .88, 1, 0.025, 0.6))

df = pd.DataFrame(data = np.linspace(begin, start, num), index = w_list, columns = ["b0"] )
print df