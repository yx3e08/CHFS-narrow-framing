import pandas as pd

for i in range(6) :
    i += 1
    i = str(i)
    vars()["inputs" + i] = pd.read_excel("inputs" + i+ ".xlsx", "Sheet1")

dff = pd.read_excel("chfsxie.xlsx", "Sheet1" )
base = pd.concat((inputs1,inputs2, inputs3, inputs4, inputs5, inputs6))
dff["narrow framing"] = base["b_list"]
dff.to_excel("dff.xlsx")
