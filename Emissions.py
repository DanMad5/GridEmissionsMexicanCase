import pandas as pd
import numpy as np
import os.path

#if __name__ == "__main__":
    
def calculateTechEFe():
    emissionFactor = [0, 0, 1416.118441, 779.810903, 345.1495515, 829.6461664, 0, 0, 0, 685.99148, 514.8325705] #emission factor for each technology for that year
    methaneFactor = [0, 0, 0, 0.010341, 0.035750, 0.239126, 0, 0, 0, 0.008935, 0.041013]
    methaneFactor = [28.9*x for x in methaneFactor]
    N2OFactor = [0, 0, 0, 0.017114, 0.012480, 0, 0, 0, 0, 0.004772, 0.014352]
    N2OFactor = [270*x for x in N2OFactor]
    NOxFactor = [0, 0, 3.31, 5.0724, 1.358981, 10.954371, 0, 0, 0, 1.0989, 1.523214]
    NOxFactor = [8.5*x for x in NOxFactor]

    EFe = emissionFactor

    #add all emission factors to an equivalent CO2 value for all technologies
    for i in range(11):
        EFe[i] += methaneFactor[i] + N2OFactor[i] + NOxFactor[i]

    return EFe

def calculateEmissions(year,EFe,techNames):
    filePath = "Data/CompleteGeneration/CompleteBrute-" + year[2:] + '.csv' #file path for the established year
    emissions = pd.DataFrame() #data frame to store all the hourly calculated emissions

    #try to open the file, see if the file exists
    try:
        emissions = pd.read_csv(filePath) #copy values of generation to emissions, to later multiply by emission factor and
                                          #efficiency of each technology

        #multiply each column by their corresponding emission factor and efficiency
        for i in range(len(techNames)):
            emissions.iloc[:,i+1] = round(emissions.iloc[:,i+1]*EFe[i],3)

        saveFilePath = "Data/Emissions" #file path for the established year
        emissions.to_csv(os.path.join(saveFilePath,"emissions-" + year[2:] + ".csv"),index=0)

    except FileNotFoundError:
        print('The file:', filePath, 'has not yet been created.')   #if the file does not exist notify the user. You should 
                                                                    #run the Formater.py to produce the file.

def calculateEF(year, techNames):
    
    filePathGeneration = "Data/CompleteGeneration/CompleteBrute-" + year[2:] + ".csv"
    filePathEmissions = "Data/Emissions/emissions-" + year[2:] + ".csv"

    generation = pd.read_csv(filePathGeneration)
    emissions = pd.read_csv(filePathEmissions)

    generation["Total"] = generation[techNames].sum(axis=1)
    emissions["Total"] = emissions[techNames].sum(axis=1)

    EF = round(emissions.iloc[:,12]/generation.iloc[:,12]/1000,4)

    saveFilePath = "Data/EmissionFactors/EFe" 
    EF.to_csv(os.path.join(saveFilePath, "EFe-" + year[2:] + ".csv"),index=0)

    return EF