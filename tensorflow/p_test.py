import matplotlib.pyplot as plt
from matplotlib.dates import (YEARLY, DateFormatter, rrulewrapper, RRuleLocator, drange)
import numpy as np
import datetime
import glob

class DataManager:
    def __init__(self):
        self.MSE = list()
        self.MAE = list()
        self.SGE = list()

    def AddMSE(self, data, cycle):
        self.MSE.append([data, cycle])

    def AddMAE(self, data, cycle):
        self.MAE.append([data, cycle])

    def AddSGE(self, data, cycle):
        self.SGE.append([data, cycle])

    def FinalizeAndPrint(self):
        self.MSE.sort(key=lambda x:x[0])
        self.MAE.sort(key=lambda x:x[0])
        self.SGE.sort(key=lambda x:x[0])

        print("----------------------------")
        for x in range(5):
            print(f"{x+1}. MSE - {self.MSE[x][0]} at cycles: {self.MSE[x][1]}")
        print("----------------------------")
        for x in range(5):
            print(f"{x+1}. MAE - {self.MAE[x][0]} at cycles: {self.MAE[x][1]}")
        print("----------------------------")
        for x in range(5):
            print(f"{x+1}. SGE - {self.SGE[x][0]} at cycles: {self.SGE[x][1]}")
        print("----------------------------")

dm = DataManager()
dm.AddMAE(1,22)
dm.AddMAE(99,23)
dm.AddMAE(1111,26)
dm.AddMAE(9,2)
dm.AddMAE(19,7)
dm.AddMAE(2,5)

dm.AddMSE(1,22)
dm.AddMSE(29,23)
dm.AddMSE(111,26)
dm.AddMSE(11,2)
dm.AddMSE(19,7)
dm.AddMSE(2,5)

dm.AddSGE(10,22)
dm.AddSGE(909,23)
dm.AddSGE(10111,26)
dm.AddSGE(90,2)
dm.AddSGE(109,7)
dm.AddSGE(20,5)

dm.FinalizeAndPrint()