
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
from matplotlib import pyplot

import numpy
from numpy import array
from numpy import hstack
import matplotlib.pyplot as plt
from pandas import read_csv
import datetime
import glob, os
import math
import random
import sys, getopt
# from keras.optimizers import SGD
# from keras.models import Sequential
# from keras.models import load_model
# from keras.callbacks import Callback
# from keras.layers import Dense
# from keras.layers import Dropout
# from keras.layers import LSTM
# from keras.layers import RepeatVector
# from keras.layers import TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error

class DataManager:
    def __init__(self):
        self.MSE = list()
        self.MAE = list()
        self.SLE = list()

    def AddMSE(self, data, cycle):
        self.MSE.append([data, cycle])

    def AddMAE(self, data, cycle):
        self.MAE.append([data, cycle])

    def AddSLE(self, data, cycle):
        self.SLE.append([data, cycle])

    def Reset(self):
        self.MSE.clear()
        self.MAE.clear()
        self.SLE.clear()

    def Write(self, fileName, lines):
        if (fileName == ""):
            for x in lines:
                print(x, end="")
        else:
            f = open(fileName+".txt", "w")
            for x in lines:
                f.write(x)
            f.close()

    def FinalizeAndPrint(self, fileName):
        self.MSE.sort(key=lambda x:x[0])
        self.MAE.sort(key=lambda x:x[0])
        self.SLE.sort(key=lambda x:x[0])

        lines = list()
        numToWrite = 10
        
        lines.append("----------------------------\n")
        r_mse = 0
        if (len(self.MSE)>numToWrite):
            r_mse = numToWrite
        else:
            r_mse = len(self.MSE)
        for x in range(r_mse):
            lines.append(f"{x+1}. MSE - {self.MSE[x][0]} at cycles: {self.MSE[x][1]}\n")

        lines.append("----------------------------\n")
        r_mae = 0
        if (len(self.MAE)>numToWrite):
            r_mae = numToWrite
        else:
            r_mae = len(self.MAE)
        for x in range(r_mae):
            lines.append(f"{x+1}. MAE - {self.MAE[x][0]} at cycles: {self.MAE[x][1]}\n")

        lines.append("----------------------------\n")
        r_sle = 0
        if (len(self.SLE)>numToWrite):
            r_sle = numToWrite
        else:
            r_sle = len(self.SLE)
        for x in range(r_sle):
            lines.append(f"{x+1}. SLE - {self.SLE[x][0]} at cycles: {self.SLE[x][1]}\n")
        lines.append("----------------------------\n")

        self.Write(fileName, lines)

def cycle_analysis(data, cycle, forecast_plot = False):
    training = []
    testing = []
    if (len(data) > 6000 ):
        training = data[-1500:-5].iloc[:-1,]
        testing = data[-5:]
    else:
        training = data[0:-200].iloc[:-1,]
        testing = data[-200:]
    predict_period = 600#len(pd.date_range(split_date,max(data.index)))
    df = training.reset_index()
    df.columns = ['index','ds','y']
    training.columns = ['ds','y']
    testing.columns = ['ds','y']
    #m = Prophet(weekly_seasonality=False,yearly_seasonality=False,daily_seasonality=False)
    #m.add_seasonality('self_define_cycle',period=cycle,fourier_order=32,mode=mode)

    m = Prophet(
    growth="linear",
    #holidays=holidays,
    seasonality_mode='multiplicative',  # mcmc-vel muiltiplicative-ot kell hasznalni --- map-pel additive-ot
    #changepoint_prior_scale=0.3,
    #seasonality_prior_scale=0.3,
    interval_width=0.8,                 # egy cycle szelessegenek rugalmassaga
    #holidays_prior_scale=20,
    mcmc_samples=60,
    #changepoint_prior_scale=30,
    #changepoint_range=0.91,
    #seasonality_prior_scale=35,
    #holidays_prior_scale=20,
    daily_seasonality=False,
    weekly_seasonality=False,
    yearly_seasonality=True,
    )
    
    for c in cycle:
        m.add_seasonality(name='spec', period=c, fourier_order=40) # prior_scale=15

    m.fit(df)
    future = m.make_future_dataframe(periods=predict_period)
    forecast = m.predict(future)
    if forecast_plot:
        m.plot(forecast)

        # trainDate = training["ds"]
        # trainValue = training["y"]
        # plt.plot(trainDate, trainValue, '.', color='#cccccc', alpha=0.6)

        testDate = testing.values[:,0] #date
        testValue = testing.values[:,1] #value

        conv_dates = []
        for i in range(len(testDate)):
            date1 = datetime.datetime.strptime(testing.values[i,0], '%Y-%m-%d').date()
            conv_dates = numpy.append(conv_dates, date1)
        plt.plot(conv_dates, testValue, '.', color='#ff3333', alpha=0.6)

        plt.xlabel('Date', fontsize=12, fontweight='bold', color='gray')
        plt.ylabel('Price', fontsize=12, fontweight='bold', color='gray')
        plt.show()
    
    temp = forecast['yhat']
    Mse = mean_squared_error(testing["y"], temp[0:len(testing["y"])] )
    Mae = mean_absolute_error(testing["y"], temp[0:len(testing["y"])] )
    Sle = mean_squared_log_error(testing["y"], temp[0:len(testing["y"])] )
    
    dm.AddMSE(Mse, cycle)
    dm.AddMAE(Mae, cycle)
    dm.AddSLE(Sle, cycle)

    print(f"MSE: {Mse} - MAE: {Mae} at cycle: {cycle}")

    return 0

def RunOneTest():
    fileName = "./data/VIX2_daily.csv"
    df = pd.read_csv(fileName, usecols=[0,4])
    print("One Test  - Processing: "+fileName)
    cycle_analysis(df, [51, 196, 127, 245, 294, 315, 365, 445, 771], forecast_plot=True) 
    #[31, 87, 127, 154, 176, 211, 245, 295, 372, 616, 682, 729, 771]
    #[51, 196, 127, 245, 294, 315, 445, 771] 
    dm.FinalizeAndPrint("")

def RunOneCycle():
    os.chdir("./data")
    for fileName in glob.glob("*.csv"):
        df = pd.read_csv(fileName, usecols=[0,4])
        dm.Reset()
        print("Search 1 parameter processing: "+fileName)
        for i in range(10,370):
            cycle_analysis(df, [i], forecast_plot=False)
        print(fileName)
        dm.FinalizeAndPrint(fileName+"1")

def RunTwoCycle():
    os.chdir("./data")
    for fileName in glob.glob("*.csv"):
        df = pd.read_csv(fileName, usecols=[0,4])
        dm.Reset()
        print("Search 2 parameter processing: "+fileName)
        for i in range(10,366):
            for o in range(i+1,366):
                cycle_analysis(df, [i,o], forecast_plot=False)
        print(fileName)
        dm.FinalizeAndPrint(fileName+"2")

dm = DataManager()
RunOneTest()
print("Done.")

#VIX!
#RPC maximum  data: 4.653 at cycle: 283
#MSE minimum  data: 25.494961623712165 at cycle: 282
#MAE minimum  data: 3.5114502841168234 at cycle: 282

#VIX! 2000-6500 learn, 6500-7770 test
#RPC maximum  data: 4.653 at cycle: 283
#MSE minimum  data: 25.494961623712165 at cycle: 282
#MAE minimum  data: 3.5114502841168234 at cycle: 282

#VIX! 3500:6770 learn
#RPC maximum  data: 4.852 at cycle: 279
#MSE minimum  data: 21.29917709610351 at cycle: 209
#MAE minimum  data: 3.049376297045781 at cycle: 279

#IBM  3500:6770 learn
#RPC maximum  data: 0.825 at cycle: 281
#MSE minimum  data: 2.651962963011946 at cycle: 275
#MAE minimum  data: 1.1478663267450626 at cycle: 187

#DIS  3500:6770 learn
#RPC maximum  data: 0.091 at cycle: 295
#MSE minimum  data: 0.016806448362202537 at cycle: 296
#MAE minimum  data: 0.08833902777663996 at cycle: 101

#DIS  8380:-365
#RPC maximum  data: 3.799 at cycle: 211
#MSE minimum  data: 19.819353471678923 at cycle: 274
#MAE minimum  data: 3.0310232513501534 at cycle: 286

#DIS 8015:-1105