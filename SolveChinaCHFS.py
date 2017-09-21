# -*- coding: utf-8 -*-
"""
Created on Thu Oct 06 13:33:58 2016

This file calculates the level of b0(narrow framing) using the methods developed by Barberis and Ming (2009), section 6

@author: Evojoy
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
import sympy as sy

class narrow_framing():
    def __init__(self, gamma, LA, weight_s):

        self.beta = 0.98

        #China
        self.gc = 0.079
        self.csd = 0.014723833

        self.w = 0.362621114

        self.ssd = 0.536807495
        self.rf = 1.0331
        self.gs = 0.5 * (-self.ssd ** 2.0 + 2.0 * np.log(self.rf + 0.042854))

        self.gamma = gamma
        self.ws = weight_s
        self.LA = LA
        #self.b = 0.03

        #US
        # self.gc = 0.0184
        # self.csd = 0.0379
        #
        # self.w = 0.1
        #
        # self.ssd = 0.2
        # ep = 0.0366
        # self.rf = 1.031
        # self.gs = 0.5 * (-self.ssd ** 2.0 + 2.0 * np.log(self.rf + ep))
        #
        # self.gamma = gamma
        # self.ws = weight_s
        # self.LA = LA
        #self.b = 0.02

        self.a = 1 - self.beta ** (1 / self.gamma) * self.rf ** ((1 - self.gamma) / self.gamma) * np.exp(0.5 * (1 - self.gamma) * (self.csd ** 2))
        self.es = (np.log(self.rf) - self.gs) / self.ssd

    def n(self,x):
        """
        return the cumulative probability of a standard normal variables
        """
        stand_normal = norm(0, 1)
        return stand_normal.cdf(x)

    def solver(self,b0):
        Eq14 = b0 * self.rf * (self.beta / (1 - self.beta)) ** (1 / (1 - self.gamma)) * ((1 - self.a) / self.a) ** (-self.gamma / (1 - self.gamma)) * (
            np.exp(self.gs + 0.5 * (self.ssd ** 2)) - self.rf + (self.LA - 1) * (np.exp(self.gs + 0.5 * (self.ssd ** 2)) * self.n(self.es - self.ssd) - self.rf * self.n(self.es))
        ) + np.exp(self.gs + 0.5 * (self.ssd ** 2) - self.gamma * self.ssd * self.csd * self.w) - self.rf

        Eq15 = b0 * self.rf * (self.beta / (1 - self.beta)) ** (1 / (1 - self.gamma)) * self.ws * ((1 - self.a) / self.a) ** (-self.gamma / (1 - self.gamma)) * (
            np.exp(self.gs + 0.5 * (self.ssd ** 2)) - self.rf + (self.LA - 1) * (np.exp(self.gs + 0.5 * (self.ssd ** 2)) * self.n(self.es - self.ssd) - self.rf * self.n(self.es))
        ) + (1 / (1 - self.a)) * np.exp(self.gc + 0.5 * (self.csd ** 2) - self.gamma * self.csd * self.csd) - self.rf
        #print Eq14, Eq15
        return  abs(Eq15) #+ abs(Eq14)

    def Ssolver(self,b0 = sy.symbols("b0")):
        Eq15 = b0 * self.rf * (self.beta / (1 - self.beta)) ** (1 / (1 - self.gamma)) * self.ws * ((1 - self.a) / self.a) ** (-self.gamma / (1 - self.gamma)) * (
            sy.exp(self.gs + 0.5 * (self.ssd ** 2)) - self.rf + (self.LA - 1) * (sy.exp(self.gs + 0.5 * (self.ssd ** 2)) * self.n(self.es - self.ssd) - self.rf * self.n(self.es))
        ) + (1 / (1 - self.a)) * sy.exp(self.gc + 0.5 * (self.csd ** 2) - self.gamma * self.csd * self.csd) - self.rf
        return sy.solve( (Eq15)**2 )[0]

    def get_gs(self, rf, ep):
        x = 0.5 * (-self.ssd ** 2.0 + 2.0 * np.log(rf + ep))
        return x

    def express_b(self, b):
        """
        This function express Eq15 when gs has been substituted
        """
        NN = self.n(self.es - self.ssd)
        n = self.n(self.es)
        xx = -self.rf - np.exp(self.gc + self.csd**2*(1/2 - self.gamma))/(-1 + self.a) + b * self.rf* self.ws*(-1 + 1/self.a)**(self.gamma/(-1 + self.gamma))*(self.beta/
        (1 -  self.beta)) ** (1/(1 - self.gamma))*(-self.rf + 1/np.sqrt(((-1 + 1/self.a)**(self.gamma/(1 - self.gamma)) +
        b*np.exp(self.csd* self.ssd * self.w * self.gamma)*(1 + (-1 + self.LA)*NN)* self.rf*(self.beta/(1 - self.beta))**(1/(1 - self.gamma)))**2/(np.exp(2*self.csd *
        self.ssd * self.w* self.gamma)*(self.rf**2*((-1 + 1/self.a)**(self.gamma/(1 - self.gamma)) + b*(1 + (-1 + self.LA)*n)*
        self.rf*(self.beta/(1 - self.beta))**(1/(1 - self.gamma)))**2))) + (-1 + self.LA)*((-n)*self.rf + NN/np.sqrt(((-1 + 1/self.a)**(self.gamma/(1 - self.gamma)) + b*
        np.exp(self.csd* self.ssd * self.w * self.gamma)*(1 + (-1 + self.LA)*NN)*self.rf*(self.beta/(1 - self.beta))**(1/(1 - self.gamma)))**2/(np.exp(2*self.csd * self.ssd*
        self.w *self.gamma)*(self.rf**2*((-1 + 1/self.a)**(self.gamma/(1 - self.gamma)) + b*(1 + (-1 + self.LA)*n)*self.rf*(self.beta/(1 - self.beta))**
        (1/(1 - self.gamma)))**2)))))
        return xx

    def express_rf(self, rf):
        """
        This function express Eq15 when gs has been substituted
        """
        NN = self.n(self.es - self.ssd)
        n = self.n(self.es)
        xx = -rf - np.exp(self.gc + self.csd**2*(1/2 - self.gamma))/(-1 + self.a) + self.b*rf* self.ws*(-1 +
        1/self.a)**(self.gamma/(-1 + self.gamma))*(self.beta/(1 -  self.beta)) ** (1/(1 - self.gamma))*(-rf + 1/np.sqrt(((-1 + 1/self.a)**(self.gamma/(1 - self.gamma)) + self.b*np.exp(self.csd* self.ssd * self.w * self.gamma)*(1 + (-1 + self.LA)*NN)*
        rf*(self.beta/(1 - self.beta))**(1/(1 - self.gamma)))**2/(np.exp(2*self.csd * self.ssd * self.w* self.gamma)*(rf**2*((-1 +
        1/self.a)**(self.gamma/(1 - self.gamma)) + self.b*(1 + (-1 + self.LA)*n)* rf*(self.beta/(1 - self.beta))**(1/(1 - self.gamma)))**2))) + (-1 + self.LA)*((-n)*rf +
        NN/np.sqrt(((-1 + 1/self.a)**(self.gamma/(1 - self.gamma)) + self.b*np.exp(self.csd* self.ssd * self.w * self.gamma)*(1 + (-1 + self.LA)*NN)*
        rf*(self.beta/(1 - self.beta))**(1/(1 - self.gamma)))**2/(np.exp(2*self.csd * self.ssd* self.w *self.gamma)*(rf**2*((-1 + 1/self.a)**(self.gamma/(1 - self.gamma))
        + self.b*(1 + (-1 + self.LA)*n)* rf*(self.beta/(1 - self.beta))**(1/(1 - self.gamma)))**2)))))
        return xx

    def find_b(self):
        """
        This file estimates b when setting Eq15 equals to zero and rf is known
        """
        rlist = []
        b_list = np.linspace(0, 1.25, 1251)
        j = 0
        for q in b_list:
            rlist.append(self.solver(q))
            #rlist.append(self.express_b(q))
            #print self.express_b(q)
            #print j
            j += 1
        # Find the element of ID that returns the lowest values of Eq14 and Eq15
        #id0 = min(enumerate(rlist), key=lambda x: abs(x[1] - 0))[0]
        id0 = min(enumerate(rlist), key=lambda x: abs(x[1] - 0))[0]
        #print id0
        return round(b_list[id0], 4)

    def find_rf(self):
        """
        This file estimates rf when setting Eq15 equals to zero and b is known
        """
        rf_list  = []
        rf_candidate = np.linspace(1,1.1,101)
        j = 0
        for i in rf_candidate:
            rf_list.append(self.express_rf(i))
            #rf_list.append(self.express_rf(i))
            print self.express_rf(i)
            print j
            j += 1
        id0 = min(enumerate(rf_list), key=lambda x: abs(x[1] - 0))[0]
        print id0
        return round(rf_candidate[id0],4)
#
# x = narrow_framing(1.5, 0.3, 2.0)
# y = x.find_b()
# z = x.find_rf()
# print y

# x = narrow_framing(7.5, 0.17868, 3.5)
# y = x.find_b()
# print y


def read_excel():
    return pd.read_excel("C:\Dropbox/Research Project/CHFS-Research/chfsxie.xlsx", "Sheet1" )
    #return pd.read_excel("D:/OneDrive/school/research/CHFSstudy/7narrow_framing/data_processed/chfsxie.xlsx", "Sheet1")
b_list = []
df = read_excel()
#print df


#you can also add index into this loop
# here you must
kk = 0
for row in df.iterrows():
    kk +=1
    print kk
    gamma = float((row[1])[4])
    #LA = float((row[1])[12])
    LA = 2.25
    ws = float((row[1])[48]) # here you must be very careful when new variables are added, which requires a immediate adjustment for the loc of ws
    print gamma, LA, ws
    x = narrow_framing(gamma, LA, ws)
    #b = x.find_b()
    b1 = x.Ssolver()
    b_list.append(b1)
    #print b1

df["b_list"] = b_list

df.to_excel("df_equal_la.xlsx")
