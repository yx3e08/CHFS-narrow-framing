import pandas as pd
import statsmodels.api as sm
import statsmodels as sms
import numpy as np
# load the datasheet that contains b values
def read_excel():
    #return pd.read_excel("C:/Dropbox\Research Project/CHFS-Research/df_equal_la.xlsx","Sheet1") # this uses narrow framing when loss avesion is the same for all households
    return pd.read_excel("C:/Dropbox\Research Project/CHFS-Research/dff.xlsx", "Sheet1")
    #return pd.read_excel("D:/OneDrive/school/research/CHFSstudy/7narrow_framing/data_processed/df.xlsx", "Sheet1")

df = read_excel()
df =df[df["narrow framing"]>0]

# regroup, robustness test
#df = df[df["interested in finance"] > 4]
#df = df[df["hhead_edugrp_3"] == 0]
#df = df[df["financial_literacy1"] > 1]
#df = df[df["stock_year"]<3]

# drop unnecessary columns befere running regressions
# prepare a few regerssions with different independents

ns_4 = df[['narrow framing', 'netwealth', 'total_income_imp',
           'hhead_age', 'hhead_male','hhead_married','hhead_edugrp_3',
           'interested in finance', 'dummy_financial_job', 'financial_literacy1', 'stock_year',
           'trust', 'herding', 'ambiguity aversion']] # add behavioral characters
# interactions

#ns_4["interaction"]= ns_4["stock_year"] * ns_4["narrow framing"]
#ns_4["interaction"]= ns_4["interested in finance"] * ns_4["narrow framing"]
#ns_4["interaction"]= ns_4["hhead_edugrp_3"] * ns_4["narrow framing"]

#add a constant abd Newey-West standard errors with 12 lags.

def run(X1):
    ns_4["Interaction"] = X1 * ns_4["narrow framing"] 
    model = sm.OLS(((df["Nstock"])), sm.add_constant(ns_4)).fit() 
    print model.summary()
    output = pd.concat((model.params,model.tvalues), axis = 1)
    output.columns = ["coef","t_statistic"]
    output.round(3)
    outputs = output.transpose()

    outputs = outputs.rename(columns = {'narrow framing':'NF', "netwealth": "NW", "total_income_imp" : "FI",
                                "hous_yes":"PP", "hhead_age":"AGE","hhead_male":"GD", "hhead_married" : "ME", "hhead_edugrp_3" : "CE",
                                 "interested in finance": "IF", "Deversity" : "DY", "dummy_financial_job": "FS","financial_literacy1":"FL",
                                "stock_year": "IE", "trust": "TT", "herding": "HD", "ambiguity aversion":"AA"}) 
    outputs = outputs.transpose()
    outputs_list_4 = []
    for i in range(len(outputs)):
        outputs_list_4.append(outputs["coef"][i])
        outputs_list_4.append(outputs["t_statistic"][i])
    
    new_index = []
    for i in range(len(outputs.index)):
        new_index.append(outputs.index[i])
        new_index.append(" ")
    
    outputs_list = pd.DataFrame(outputs_list_4)
    outputs_list.index = new_index
    #Table.to_excel("Table.xlsx")
    #outputs_list.to_excel("interation.xlsx")
    return  outputs_list
    
#a = pd.concat( (run(ns_4["stock_year"]), run(ns_4["interested in finance"]), run(ns_4["hhead_edugrp_3"])), axis = 1)
a = pd.concat( (run(ns_4["hhead_married"]), run(ns_4["ambiguity aversion"]), run(ns_4["hhead_edugrp_3"])), axis = 1)

a.columns = ["IE", "IF", "CE"]
a = a.round(3)
#a.to_excel("interation.xlsx")

