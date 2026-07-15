import pandas as pd
import os.path
import Adjustment
import Emissions
import Figures
import StateEmissionsMap

def meanEF(EF, name):
    EF = round(EF.mean(axis=0),4)
    EF.name = name
    EF.to_frame()
    EF.index = yearRange
    return EF

techNames = ['EO','FV','BIO','CE','CC','CI','GEO','HE','NU','TC','TG']
yearRange = range(2017,2025)

EF = Emissions.calculateTechEFe()

print("Emission Factors used are: \n", techNames, "\n", EF, "\n")

for year in yearRange:
    year = str(year)
    Emissions.calculateEmissions(year,EF,techNames)
print("Emissions updated from years 2017-2024. \n")

allYearsEFe = pd.DataFrame()
allYearsEFeMA = pd.DataFrame()
allYearsEFeAdj = pd.DataFrame()
allYearsEFeMAAdj = pd.DataFrame()

for year in yearRange:
    year = str(year)
    
    EFe = pd.DataFrame()
    EFe["EFe"] = Emissions.calculateEF(year, techNames)
    allYearsEFe = pd.concat([allYearsEFe,EFe["EFe"]], axis=1)
    
    EFeMA = pd.DataFrame()
    name = "EFe-MA"
    EFeMA[year] = Adjustment.movingAverage(EFe, year, name) 
    allYearsEFeMA = pd.concat([allYearsEFeMA,EFeMA[year]], axis=1)

    EFeAdj = pd.DataFrame()
    name = "EF-Adj"
    EFeAdj[year] = Adjustment.linearRegression(year, name, techNames)
    allYearsEFeAdj = pd.concat([allYearsEFeAdj,EFeAdj[year]], axis=1)

    EFeMAAdj = pd.DataFrame()
    name = "EF-MA-Adj"
    EFeMAAdj[year] = Adjustment.movingAverage(EFeAdj, year, name)
    allYearsEFeMAAdj = pd.concat([allYearsEFeMAAdj,EFeMAAdj[year]], axis=1)
    
allEFeYear = meanEF(allYearsEFe,"EFe")
allEFeYear = pd.concat([allEFeYear,meanEF(allYearsEFeMA,"EFe-MA")], axis=1)
allEFeYear = pd.concat([allEFeYear,meanEF(allYearsEFeAdj,"EFe-Adj")], axis=1)
allEFeYear = pd.concat([allEFeYear,meanEF(allYearsEFeMAAdj,"EFe-MA-Adj")], axis=1)

print("EFe are: \n",allEFeYear, "\n")

#Fig 2
Figures.emissionFactorYears()
#Fig 3
Figures.boxAndViolin()
#Fig 4
Figures.heatmap()
#Fig 5
Figures.energyTech()
#Fig 6 
Figures.capacityFactor()
#Fig 7 
Figures.capacityFactorDistribution()
#Fig 8 map
StateEmissionsMap.generateEmissionsMap()
#Fig 9 Percentage Change
Figures.growth_by_technology()
#Fig 10 XGBoost
Figures.emissionFactorForecast()
