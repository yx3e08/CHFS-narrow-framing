import pandas as pd
import numpy as np

#path = "D:/OneDrive/school/research/CHFSstudy/7narrow_framing/data_processed/chfs9.xls"
path = "C:\Dropbox/Research Project/CHFS-Research/chfs9.xls"
chfs = pd.read_excel(path, "Sheet1", skiprows = 0)
chfs["s_debt"] = chfs["s_debt"].fillna(0)
chfs["cash_stock_account"] = chfs["cash_stock_account"].fillna(0)
chfs = chfs[chfs['Nstock'] > 0]

chfs = chfs[chfs['risk aversion'] < 6]

chfs = chfs[chfs['risk aversion'] > 0]

chfs = chfs[chfs['netwealth'] > 0]

chfs = chfs[chfs['stock'] > 0]
chfs = chfs[chfs["s_debt"] >= 0]
chfs = chfs[chfs['cash_stock_account'] >= 0]

chfs["netstock"] = chfs["stock"] + chfs["cash_stock_account"] + chfs["stock_in_fund"]- chfs["s_debt"]
#chfs["netstock"] = chfs["stock"] + chfs["cash_stock_account"] - chfs["s_debt"]
chfs = chfs[chfs['netstock'] > 0]
chfs = chfs[chfs['loss aversion'] > 0]

chfs = chfs[chfs["a4002a"] >=0]
chfs = chfs[chfs["d3102d"] >=0]
chfs = chfs[chfs["d3111d"] >=0]
chfs = chfs[chfs["hhead_edu_years"] >=0]
chfs = chfs[chfs["hhead_edugrp_1"] >=0]
chfs = chfs[chfs["hhead_edugrp_2"] >=0]
chfs = chfs[chfs["hhead_edugrp_3"] >=0]


chfs["w_stock"] = chfs["netstock"]/chfs["netwealth"]
chfs["risk aversion"] = (chfs["risk aversion"] + .5)/2 +.75 + .5
chfs["loss aversion"] = -chfs["loss aversion"] + 4

df = chfs
# rename some variables to make them more intuitive
df=df.rename(columns = {'a4002a':'interested in finance', "a4002b": "finance class", "insurance_dummy" : "ambiguity aversion",
                        "d3102d":"trust", "d3111d":"herding","b_list":"narrow framing"})

# clean independent variables
df = df[df["total_income_imp"] > 0]
df["total_income_imp"] = np.log(df["total_income_imp"])
df = df[df["Nstock"] <= 30] # control outliers if it exceeds 3 times of standard deviation
df = df[df["w_stock"] > 0.05]
df = df[df["w_stock"] < 1]
df = df[df["netstock"] > 10000]
df = df[df["netwealth"] > 100000]

df["trust"] = -1*(df["trust"]-6) # resort the trust scroes to make it make it more itunitive (larger values represent more credible).
df["herding"] = -1*(df["herding"]-6) # resort the herding scores to make it more itunitive (larger values represent more likley to follow others).
df["interested in finance"] = -1*(df["interested in finance"]-6) # resort the scores to make it more itunitive (larger values represent more interested in finance).
df["ambiguity aversion"] = ((-(df["loss aversion"]-4)) - 1) + df["ambiguity aversion"] #
df["hhead_age"] = np.log(df["hhead_age"])
# scale down some variables
df["netwealth"] = np.log(df["netwealth"])
df["total_income_imp"] = np.log(df["total_income_imp"])
df = df[df["hous_ratio"] >= 0]
df = df[df["hous_ratio"] < 1]
df = df[df["hh_size"] >= 0]


# work out investment property
a = df["hh_size"] - 1
a[a>0] = 1
df["hsize"] = a


df["IP"] = df["hous_ratio"]
#print (df)
df["IP"][df["IP"]>=0.7] = 1
df["IP"][df["IP"]<0.7] = 0
#df["IP"] = b

#print chfs.shape
#df.to_excel("chfsxie.xlsx")