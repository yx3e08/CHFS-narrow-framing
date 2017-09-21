import pandas as pd
import statsmodels.api as sm
import numpy as np
# load the datasheet that contains b values
def read_excel():
    #return pd.read_excel("C:/Dropbox\Research Project/CHFS-Research/df_equal_la.xlsx","Sheet1")  # this uses narrow framing when loss ave
    return pd.read_excel("C:/Dropbox\Research Project/CHFS-Research/df.xlsx", "Sheet1")
    #return pd.read_excel("D:/OneDrive/school/research/CHFSstudy/7narrow_framing/data_processed/df.xlsx", "Sheet1")

df = read_excel()
# rename some variabels to make them more intunitive
df=df.rename(columns = {'a4002a':'interested in finance', "a4002b": "finance class", "insurance_dummy" : "ambiguity aversion",
                        "d3102d":"trust", "d3111d":"herding","b_list":"narrow framing"})

# clean independent variables
df = df[df["total_income_imp"] > 0]
# control the outliers and make subsets

#df = df[df["interested in finance"] > 3]
#df = df[df["hhead_edugrp_3"] == 0]
#df = df[df["ambiguity aversion"] == 0]

df["total_income_imp"] = np.log(df["total_income_imp"])
df = df[df["Nstock"] <= 30] # control outliers if it exceeds 3 times of standard deviation
df = df[df["w_stock"] > 0]
df = df[df["w_stock"] < 1]
df = df[df["netstock"] > 10000]
df = df[df["netwealth"] > 100000]
df = df[df["netwealth"] > 100000]

# finance knowledge

#df = df[df["interested in finance"] <= 1] # median is 3
#df = df[df["finance class"] == 1] # 1 is yes, 2 is no

# overconfidence

#df = df[df["stock_year"] < 3]
#df = df[df["hhead_male"]  == 0]
#df = df[df["hhead_married"]  == 0]

# scale down some variables
df["netwealth"] = df["netwealth"]/1000000
df["fasset"] = df["fasset"]/1000000
#df["netstock"] = df["netstock"]/1000000

#  investor sophistication
#df = df[df["total_income_imp"] < 12.782566957316348]
#df = df[df["hhead_age"] > 40]

#df["mm"] = df["trust"] * df["herding"]

# drop unnecessary columns befere running regressions
X1 = df.drop(["sid","risk aversion", "loss aversion", "cash_stock_account","Nstock", "s_debt", "stock", "hh_size",
              "prov_code", "region", "west", "middle", "rural", "swgt", "stock_in_fund", "financial_literacy",
              "hhead_edugrp_1", "hhead_edugrp_2", "hhead_edugrp", "hhead_edu",
              "hhead_edu_years", "w_stock", "health", "busiyes", "happiness", "htotal_worker_number","netstock"
              ],axis = 1)

#X1["herding"] = -X1["interested in finance"] + 6
X1["stock_year * narrow_framing"] = X1["hhead_edugrp_3"] * X1["narrow framing"]

#edit this DF from here to have sub-regressions
#add a constant
model = sm.OLS(np.log(df["Nstock"]), sm.add_constant(X1)).fit() #log can increase the significance sharply

#X = df[['hhead_married', "herding", "trust", "hhead_male", "fasset"]]
#model2 = sm.OLS(np.log(df["Nstock"]), sm.add_constant(X)).fit() #log can increase the significance sharply

print model.summary()

#print model2.summary()

# outputs = pd.concat((model.params,model.bse,model.pvalues), axis = 1)
# outputs.columns = ["coef", "standard errors", "p-values"]
# outputs = outputs.round(4)
#
# #export to an excel datasheet
# outputs.to_excel("regresion_b_nstock.xlsx")

