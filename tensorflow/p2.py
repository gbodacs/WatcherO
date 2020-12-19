
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

    def FinalizeAndPrint(self, fileName):
        self.MSE.sort(key=lambda x:x[0])
        self.MAE.sort(key=lambda x:x[0])
        self.SLE.sort(key=lambda x:x[0])

        f = open(fileName+".txt", "w")

        f.write("----------------------------")
        for x in range(5):
            f.write(f"{x+1}. MSE - {self.MSE[x][0]} at cycles: {self.MSE[x][1]}")
        f.write("----------------------------")
        for x in range(5):
            f.write(f"{x+1}. MAE - {self.MAE[x][0]} at cycles: {self.MAE[x][1]}")
        f.write("----------------------------")
        for x in range(5):
            f.write(f"{x+1}. SLE - {self.SLE[x][0]} at cycles: {self.SLE[x][1]}")
        f.write("----------------------------")
        f.close()

dm = DataManager()

def cycle_analysis(data, cycle, mode='additive', forecast_plot = False):
    training = data[0:-300].iloc[:-1,]
    testing = data[-300:]
    predict_period = 1500#len(pd.date_range(split_date,max(data.index)))
    df = training.reset_index()
    df.columns = ['index','ds','y']
    #m = Prophet(weekly_seasonality=False,yearly_seasonality=False,daily_seasonality=False)
    #m.add_seasonality('self_define_cycle',period=cycle,fourier_order=32,mode=mode)

    # sort by salary (Descending order)
    employees.sort(key=get_salary, reverse=True)
    print(employees, end='\n\n')

    m = Prophet(
    growth="linear",
    #holidays=holidays,
    seasonality_mode=mode,
    changepoint_prior_scale=30,
    seasonality_prior_scale=35,
    holidays_prior_scale=20,
    daily_seasonality=False,
    weekly_seasonality=False,
    yearly_seasonality=False,
    ).add_seasonality(
        name='spec',
        period=cycle,
        fourier_order=32
    #).add_seasonality(
    #    name='daily',
    #    period=1,
    #    fourier_order=15
    # ).add_seasonality(
    #     name='weekly',
    #     period=7,
    #     fourier_order=20
    # ).add_seasonality(
    #     name='yearly',
    #     period=365.25,
    #     fourier_order=20
    )
    # .add_seasonality(
    #     name='quarterly',
    #     period=365.25/4,
    #     fourier_order=32,
    #     prior_scale=15)

MIN_mse = SaveData(20000, 0)
MIN_mae = SaveData(20000, 0)

def cycle_analysis(data, split_date, cycle,mode='additive', forecast_plot = False):
    training = data[2000:6500].iloc[:-1,]
    testing = data[6500:]
    predict_period = len(pd.date_range(split_date,max(data.index)))
    df = training.reset_index()
    df.columns = ['index','ds','y']
    m = Prophet(weekly_seasonality=False,yearly_seasonality=False,daily_seasonality=False)
    m.add_seasonality('self_define_cycle',period=cycle,fourier_order=8,mode=mode)
    m.fit(df)
    future = m.make_future_dataframe(periods=predict_period)
    forecast = m.predict(future)
    if forecast_plot:
        m.plot(forecast)
        plt.plot(testing.index,testing.values,'.',color='#ff3333',alpha=0.6)
        plt.xlabel('Date',fontsize=12,fontweight='bold',color='gray')
        plt.ylabel('Price',fontsize=12,fontweight='bold',color='gray')
        m.plot_components(forecast)
        plt.show()
    
    temp = forecast['yhat']
    Mse = mean_squared_error(training["y"], temp[0:len(training["y"])] )
    Mae = mean_absolute_error(training["y"], temp[0:len(training["y"])] )
    Sle = mean_squared_log_error(training["y"], temp[0:len(training["y"])] )
    
    dm.AddMSE(Mse, cycle)
    dm.AddMAE(Mae, cycle)
    dm.AddSLE(Sle, cycle)

    print(f"MSE: {Mse} - MAE: {Mae} at cycle: {cycle}")

    return 0


os.chdir("./data")
for fileName in glob.glob("*.csv"):
    df = pd.read_csv(fileName, usecols=[0,4])
    dm.Reset()
    for i in range(10,370):
        cycle_analysis(df, i, 'additive', forecast_plot=False)
    print(fileName)
    dm.FinalizeAndPrint(fileName)

print("Done.")

#VIX!
#RPC maximum  data: 4.653 at cycle: 283
#MSE minimum  data: 25.494961623712165 at cycle: 282
#MAE minimum  data: 3.5114502841168234 at cycle: 282