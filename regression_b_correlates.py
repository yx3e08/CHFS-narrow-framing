import pandas as pd
import statsmodels.api as sm
import numpy as np
# load the datasheet that contains b values
def read_excel():
    return pd.read_excel("C:/Dropbox\Research Project/CHFS-Research/df.xlsx", "Sheet1")
    #return pd.read_excel("D:/OneDrive/school/research/CHFSstudy/7narrow_framing/data_processed/df.xlsx", "Sheet1")

df = read_excel()

# scale down some variables
df["netwealth"] = df["netwealth"]/1000000
df["fasset"] = df["fasset"]/1000000
df["netstock"] = df["netstock"]/1000000
df["hhead_age_squared"] = (df["hhead_age"])**2


NF = df["narrow framing"].values

# drop unnecessary columns before running regressions
X1 = df.drop(["sid","risk aversion", "loss aversion", "cash_stock_account","Nstock", "s_debt", "stock", "hh_size", 
              "prov_code", "region", "west", "middle", "rural", "swgt", "stock_in_fund", "financial_literacy", 
              "finance class", "hhead_edugrp", "hhead_edu", "hhead_edugrp_1", "hhead_edugrp_2", "Deversity",
              "hhead_edu_years", "w_stock", "health", "busiyes", "happiness", "htotal_worker_number", "narrow framing",
              "total_income_imp", "htotal_child_number", "hhead_age_squared", "netstock",
              ],axis = 1)

#edit this DF from here to have sub-regressions
#add a constant
model = sm.OLS(np.log(NF), sm.add_constant(X1)).fit() # log can increase the significance sharply
print model.summary()

outputs = pd.concat((model.params,model.bse,model.pvalues), axis = 1)
outputs.columns = ["coef", "standard errors", "p-values"]
outputs = outputs.round(4)
outputs.to_excel("regresion_b_correlates.xlsx")