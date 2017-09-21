import pandas as pd
import statsmodels.api as sm
import statsmodels as sms
import numpy as np
# load the datasheet that contains b values
def read_excel():
    return pd.read_excel("C:/Dropbox\Research Project/CHFS-Research/dff.xlsx","Sheet1") # this uses narrow framing when loss avesion is the same for all households
    #return pd.read_excel("C:/Dropbox\Research Project/CHFS-Research/df.xlsx", "Sheet1")
    #return pd.read_excel("D:/OneDrive/school/research/CHFSstudy/7narrow_framing/data_processed/df.xlsx", "Sheet1")

df = read_excel()
df =df[df["narrow framing"]>0]
# prepare a few regerssions with different independents

ns_1 = df[['narrow framing', 'netwealth', 'total_income_imp']] # add wealth relates
ns_2 = df[['narrow framing', 'netwealth', 'total_income_imp',
           'hhead_age', 'hhead_male','hhead_married','hhead_edugrp_3']] # add demographic
ns_3 = df[['narrow framing', 'netwealth', 'total_income_imp',
           'hhead_age', 'hhead_male','hhead_married','hhead_edugrp_3',
           'interested in finance', 'dummy_financial_job', 'financial_literacy1', 'stock_year']] # add financial capacity
ns_4 = df[['narrow framing', 'netwealth', 'total_income_imp',
           'hhead_age', 'hhead_male','hhead_married','hhead_edugrp_3',
           'interested in finance', 'dummy_financial_job', 'financial_literacy1', 'stock_year',
           'trust', 'herding', 'ambiguity aversion']] # add behavioral characters
   
          
diy_1 = df[['narrow framing', 'netwealth', 'total_income_imp']] # add wealth relates
diy_2 = df[['narrow framing', 'netwealth', 'total_income_imp',
           'hhead_age', 'hhead_male','hhead_married','hhead_edugrp_3']] # add demographic
diy_3 = df[['narrow framing', 'netwealth', 'total_income_imp',
           'hhead_age', 'hhead_male','hhead_married','hhead_edugrp_3',
           'interested in finance', 'dummy_financial_job', 'financial_literacy1']] # add financial capacity           
diy_4 = df[['narrow framing', 'netwealth', 'total_income_imp',
           'hhead_age', 'hhead_male','hhead_married','hhead_edugrp_3',
           'interested in finance', 'dummy_financial_job', 'financial_literacy1',
           'trust', 'herding', 'ambiguity aversion']] # add behavioral characters
            

pp_1 = df[['narrow framing', 'netwealth', 'total_income_imp']] # add wealth relates
pp_2 = df[['narrow framing', 'netwealth', 'total_income_imp',
           'hhead_age', 'hhead_male','hhead_married','hhead_edugrp_3']] # add demographic
pp_3 = df[['narrow framing', 'netwealth', 'total_income_imp',
           'hhead_age', 'hhead_male','hhead_married','hhead_edugrp_3',
           'interested in finance', 'dummy_financial_job', 'financial_literacy1']] # add financial capacity           
pp_4 = df[['narrow framing', 'netwealth', 'total_income_imp',
           'hhead_age', 'hhead_male','hhead_married','hhead_edugrp_3',
           'interested in finance', 'dummy_financial_job', 'financial_literacy1',
           'trust', 'herding', 'ambiguity aversion']] # add behavioral characters

#model =sms.discrete.discrete_model.Probit(df["IP"], sm.add_constant(pp_4)).fit()
#print model.summary()

#model =sm.Logit(df["hous_yes"], sm.add_constant(pp_4)).fit()
#print model.summary()

#model = sm.OLS(df["hous_ratio"], sm.add_constant(pp_4)).fit() #log can increase the significance sharply
#print (model.summary())

def make_pp_4():
        
    model = sms.discrete.discrete_model.Probit(df["hsize"], sm.add_constant(pp_4)).fit()
    print (model.summary())
    outputs = pd.concat((model.params,model.tvalues), axis = 1)
    outputs.columns = ["coef","t_statistic"]
    outputs_list_2 = []
    for i in range(len(outputs)):
        outputs_list_2.append(outputs["coef"][i])
        outputs_list_2.append(outputs["t_statistic"][i])

    model = sm.OLS(df["hous_ratio"], sm.add_constant(pp_4)).fit()
    print (model.summary())
    outputs = pd.concat((model.params,model.tvalues), axis = 1)
    outputs.columns = ["coef","t_statistic"]
    outputs_list_3 = []
    for i in range(len(outputs)):
        outputs_list_3.append(outputs["coef"][i])
        outputs_list_3.append(outputs["t_statistic"][i])
        
    model = sm.OLS(df["Deversity"], sm.add_constant(pp_4)).fit() #log can increase the significance sharply
    print (model.summary())
    outputs = pd.concat((model.params,model.tvalues), axis = 1)
    outputs.columns = ["coef","t_statistic"]
    outputs_list_4 = []
    for i in range(len(outputs)):
        outputs_list_4.append(outputs["coef"][i])
        outputs_list_4.append(outputs["t_statistic"][i])

    new_index = []
    for i in range(len(outputs.index)):
        new_index.append(outputs.index[i])
        new_index.append(" ")
    Table = [outputs_list_2, outputs_list_3, outputs_list_4]
    Table = pd.DataFrame(Table).transpose()
    Table.index = new_index
    Table.columns = ["hsize", "hous_ratio", "diversity"]
    Table = (Table.fillna(" ")).round(3)
    Table = Table.rename(index = {'narrow framing':'NF', "netwealth": "NW", "total_income_imp" : "FI",
                                "hous_yes":"PP", "hhead_age":"AGE","hhead_male":"GD", "hhead_married" : "ME", "hhead_edugrp_3" : "CE",
                                 "interested in finance": "IF", "Deversity" : "DY", "dummy_financial_job": "FS","financial_literacy1":"FL",
                                "stock_year": "IE", "trust": "TT", "herding": "HD", "ambiguity aversion":"AA"}) 
    Table.to_excel("Table.xlsx")
    
    
def make_pp():
    model = sms.discrete.discrete_model.Probit(df["hous_ratio"], sm.add_constant(pp_1)).fit()
    print (model.summary())
    outputs = pd.concat((model.params,model.bse), axis = 1)
    outputs.columns = ["coef","t_statistic"]
    outputs_list_1 = []
    for i in range(len(outputs)):
        outputs_list_1.append(outputs["coef"][i])
        outputs_list_1.append(outputs["t_statistic"][i])

    model = sms.discrete.discrete_model.Probit(df["hous_ratio"], sm.add_constant(pp_2)).fit()    
    print (model.summary())
    outputs = pd.concat((model.params,model.bse), axis = 1)
    outputs.columns = ["coef","t_statistic"]
    outputs_list_2 = []
    for i in range(len(outputs)):
        outputs_list_2.append(outputs["coef"][i])
        outputs_list_2.append(outputs["t_statistic"][i])
    model = sms.discrete.discrete_model.Probit(df["hsize"], sm.add_constant(pp_3)).fit()
    print (model.summary())
    outputs = pd.concat((model.params,model.bse), axis = 1)
    outputs.columns = ["coef","t_statistic"]
    outputs_list_3 = []
    for i in range(len(outputs)):
        outputs_list_3.append(outputs["coef"][i])
        outputs_list_3.append(outputs["t_statistic"][i])

    model = sms.discrete.discrete_model.Probit(df["hous_ratio"], sm.add_constant(pp_4)).fit()
    print (model.summary())
    outputs = pd.concat((model.params,model.bse), axis = 1)
    outputs.columns = ["coef","t_statistic"]
    outputs_list_4 = []
    for i in range(len(outputs)):
        outputs_list_4.append(outputs["coef"][i])
        outputs_list_4.append(outputs["t_statistic"][i])

    new_index = []
    for i in range(len(outputs.index)):
        new_index.append(outputs.index[i])
        new_index.append(" ")
    Table = [outputs_list_1,outputs_list_2, outputs_list_3, outputs_list_4]
    Table = pd.DataFrame(Table).transpose()
    Table.index = new_index
    Table.columns = ["(1)","(2)","(3)", "(4)"]
    Table = (Table.fillna(" ")).round(3)
    Table.to_excel("Table.xlsx")
    
def make_pp_fraction():
    model = sm.OLS(df["hous_ratio"], sm.add_constant(pp_1)).fit()
    print (model.summary())
    outputs = pd.concat((model.params,model.bse), axis = 1)
    outputs.columns = ["coef","t_statistic"]
    outputs_list_1 = []
    for i in range(len(outputs)):
        outputs_list_1.append(outputs["coef"][i])
        outputs_list_1.append(outputs["t_statistic"][i])

    model = sm.OLS(df["hous_ratio"], sm.add_constant(pp_2)).fit()
    print (model.summary())
    outputs = pd.concat((model.params,model.bse), axis = 1)
    outputs.columns = ["coef","t_statistic"]
    outputs_list_2 = []
    for i in range(len(outputs)):
        outputs_list_2.append(outputs["coef"][i])
        outputs_list_2.append(outputs["t_statistic"][i])
    model = sm.OLS(df["hous_ratio"], sm.add_constant(pp_3)).fit()
    print (model.summary())
    outputs = pd.concat((model.params,model.bse), axis = 1)
    outputs.columns = ["coef","t_statistic"]
    outputs_list_3 = []
    for i in range(len(outputs)):
        outputs_list_3.append(outputs["coef"][i])
        outputs_list_3.append(outputs["t_statistic"][i])

    model = sm.OLS(df["hous_ratio"], sm.add_constant(pp_4)).fit()
    print (model.summary())
    outputs = pd.concat((model.params,model.bse), axis = 1)
    outputs.columns = ["coef","t_statistic"]
    outputs_list_4 = []
    for i in range(len(outputs)):
        outputs_list_4.append(outputs["coef"][i])
        outputs_list_4.append(outputs["t_statistic"][i])

    new_index = []
    for i in range(len(outputs.index)):
        new_index.append(outputs.index[i])
        new_index.append(" ")
    Table = [outputs_list_1,outputs_list_2, outputs_list_3, outputs_list_4]
    Table = pd.DataFrame(Table).transpose()
    Table.index = new_index
    Table.columns = ["(1)","(2)","(3)", "(4)"]
    Table = (Table.fillna(" ")).round(3)
    Table.to_excel("Table.xlsx")


def make_diy():
    model = sm.OLS(df["Deversity"], sm.add_constant(diy_1)).fit() #log can increase the significance sharply
    print (model.summary())
    outputs = pd.concat((model.params,model.bse), axis = 1)
    outputs.columns = ["coef","t_statistic"]
    outputs_list_1 = []
    for i in range(len(outputs)):
        outputs_list_1.append(outputs["coef"][i])
        outputs_list_1.append(outputs["t_statistic"][i])

    model = sm.OLS(df["Deversity"], sm.add_constant(diy_2)).fit() #log can increase the significance sharply
    print (model.summary())
    outputs = pd.concat((model.params,model.bse), axis = 1)
    outputs.columns = ["coef","t_statistic"]
    outputs_list_2 = []
    for i in range(len(outputs)):
        outputs_list_2.append(outputs["coef"][i])
        outputs_list_2.append(outputs["t_statistic"][i])

    model = sm.OLS(df["Deversity"], sm.add_constant(diy_3)).fit() #log can increase the significance sharply
    print (model.summary())
    outputs = pd.concat((model.params,model.bse), axis = 1)
    outputs.columns = ["coef","t_statistic"]
    outputs_list_3 = []
    for i in range(len(outputs)):
        outputs_list_3.append(outputs["coef"][i])
        outputs_list_3.append(outputs["t_statistic"][i])

    model = sm.OLS(df["Deversity"], sm.add_constant(diy_4)).fit() #log can increase the significance sharply
    print (model.summary())
    outputs = pd.concat((model.params,model.bse), axis = 1)
    outputs.columns = ["coef","t_statistic"]
    outputs_list_4 = []
    for i in range(len(outputs)):
        outputs_list_4.append(outputs["coef"][i])
        outputs_list_4.append(outputs["t_statistic"][i])

    new_index = []
    for i in range(len(outputs.index)):
        new_index.append(outputs.index[i])
        new_index.append(" ")
    Table = [outputs_list_1,outputs_list_2, outputs_list_3, outputs_list_4]
    Table = pd.DataFrame(Table).transpose()
    Table.index = new_index
    Table.columns = ["(1)","(2)","(3)", "(4)"]
    Table = (Table.fillna(" ")).round(3)
    Table.to_excel("Table.xlsx")

def make_ns():
    model = sm.OLS((df["Nstock"]), sm.add_constant(ns_1)).fit() #log can increase the significance sharply
    print (model.summary())
    outputs = pd.concat((model.params,model.bse), axis = 1)
    outputs.columns = ["coef","t_statistic"]
    outputs_list_1 = []
    for i in range(len(outputs)):
        outputs_list_1.append(outputs["coef"][i])
        outputs_list_1.append(outputs["t_statistic"][i])

    model = sm.OLS((df["Nstock"]), sm.add_constant(ns_2)).fit() #log can increase the significance sharply
    print (model.summary())
    outputs = pd.concat((model.params,model.bse), axis = 1)
    outputs.columns = ["coef","t_statistic"]
    outputs_list_2 = []
    for i in range(len(outputs)):
        outputs_list_2.append(outputs["coef"][i])
        outputs_list_2.append(outputs["t_statistic"][i])


    model = sm.OLS((df["Nstock"]), sm.add_constant(ns_3)).fit() #log can increase the significance sharply
    print (model.summary())
    outputs = pd.concat((model.params,model.bse), axis = 1)
    outputs.columns = ["coef","t_statistic"]
    outputs_list_3 = []
    for i in range(len(outputs)):
        outputs_list_3.append(outputs["coef"][i])
        outputs_list_3.append(outputs["t_statistic"][i])

    model = sm.OLS((df["Nstock"]), sm.add_constant(ns_4)).fit() #log can increase the significance sharply
    print (model.summary())
    outputs = pd.concat((model.params,model.bse), axis = 1)
    outputs.columns = ["coef","t_statistic"]
    outputs_list_4 = []
    for i in range(len(outputs)):
        outputs_list_4.append(outputs["coef"][i])
        outputs_list_4.append(outputs["t_statistic"][i])

    new_index = []
    for i in range(len(outputs.index)):
        new_index.append(outputs.index[i])
        new_index.append(" ")
    Table = [outputs_list_1,outputs_list_2, outputs_list_3, outputs_list_4]
    Table = pd.DataFrame(Table).transpose()
    Table.index = new_index
    Table.columns = ["(1)","(2)","(3)", "(4)"]
    Table = (Table.fillna(" ")).round(3)
    Table.to_excel("Table.xlsx")

#export to an excel datasheet
#outputs.to_excel("regresion_b_nstock_4.xlsx")
#outputs.to_excel("regresion_b_nstock.xlsx")

# export all applied variables for calculating correlation

def export_corr():
    
    ns_4["ns"] = df["Nstock"]
    ns_4["ds"] = df["Deversity"]
    ns_4["pp_dummy"] = df["hsize"]
    
    ns4 = ns_4.rename(columns = {'narrow framing':'NF', "netwealth": "NW", "total_income_imp" : "FI",
                                    "hous_yes":"PP", "hhead_age":"AGE","hhead_male":"GD", "hhead_married" : "ME", "hhead_edugrp_3" : "CE",
                                     "interested in finance": "IF", "Deversity" : "DY", "dummy_financial_job": "FS","financial_literacy1":"FL",
                                    "stock_year": "IE", "trust": "TT", "herding": "HD", "ambiguity aversion":"AA"}) 
    ns4.to_excel("corr_matix.xlsx")

#export_corr()