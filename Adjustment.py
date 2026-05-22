import os.path
import pandas as pd

def movingAverage(data, year, name):
    data = pd.DataFrame(data)
    out = round(data.rolling(window=24,min_periods=1).mean(),4)

    saveFilePath = "Data/EmissionFactors/" + name + "/"
    out.to_csv(os.path.join(saveFilePath + name +"-" + year[2:] + ".csv"),index=0)

    return out

def linearRegression(year, name, techNames):
    filePathEF = "Data/EmissionFactors/EFe/EFe-" + year[2:] + ".csv"
    filePathGeneration = "Data/CompleteGeneration/complete-" + year[2:] + ".csv"
    data = pd.read_csv(filePathEF)
    generation = pd.read_csv(filePathGeneration)
    demNet = generation[techNames].sum(axis=1).sum()/1000
    
    b0=-83.20080790236244
    b1=2.11559661e+00
    b2=4.20242038e-02
    b3=-6.09114911e-06
    out = b0 + b1*data + b2*(int(year)) + b3*demNet
    out = round(out,4)

    saveFilePath = "Data/EmissionFactors/" + name + "/"
    out.to_csv(os.path.join(saveFilePath + name + "-" + year[2:] + ".csv"),index=0)

    return out