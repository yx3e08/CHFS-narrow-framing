import numpy as np
import pandas as pd
from scipy.stats import norm
import sympy as sy
import nf2009 as nf
import time
def read_excel():
    return pd.read_excel("C:\Dropbox/Research Project/CHFS-Research/chfsxie.xlsx", "Sheet1" )
    #return pd.read_excel("D:/OneDrive/school/research/CHFSstudy/7narrow_framing/data_processed/chfsxie.xlsx", "Sheet1")
b_list = []
df = read_excel()

inputs = df[["risk aversion", "hous_ratio", "w_stock"]]
inputs = inputs[400:600]
#print(inputs)

kk = 0

start = time.clock()

for row in inputs.iterrows():
    kk +=1
    print (kk)
    gamma = round(float((row[1])[0]), 2)
    w2 = round(float((row[1])[1]), 2)
    w3 = round(float((row[1])[2]), 2)# here you must be very careful when new variables are added, which requires a immediate adjustment for the loc of ws

    print (gamma, w2, w3)
    b_list.append(nf.b_search(gamma, w2,w3))

elapsed = (time.clock() - start)
print(elapsed/60)
inputs["b_list"] = b_list
inputs.to_excel("inputs3.xlsx")