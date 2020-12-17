
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
import math
import random
import sys, getopt
from keras.optimizers import SGD
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import Callback
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

class SaveData:
    def __init__(self, data, cycle):
        self.data = data
        self.cycle = cycle

    def getData(self):
        return self.data

    def checkBigger(self, data, cycle):
        if (data>self.data):
            self.data = data
            self.cycle = cycle

    def checkSmaller(self, data, cycle):
        if (data<self.data):
            self.data = data
            self.cycle = cycle

    def setDataCycle(self, data, cycle):
        self.data = data
        self.cycle = cycle

    def printData(self, string):
        print(f"{string} data: {self.data} at cycle: {self.cycle}")

MIN_mse = SaveData(20000, 0)
MIN_mae = SaveData(20000, 0)
MAX_rpc = SaveData(0,0)

def cycle_analysis(data, split_date, cycle,mode='additive', forecast_plot = False):
    training = data[8015:-1105].iloc[:-1,]
    testing = data[-1105:]
    predict_period = 1500#len(pd.date_range(split_date,max(data.index)))
    df = training.reset_index()
    df.columns = ['index','ds','y']
    m = Prophet(weekly_seasonality=False,yearly_seasonality=False,daily_seasonality=False)
    m.add_seasonality('self_define_cycle',period=cycle,fourier_order=32,mode=mode)
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
        for i in range(len(testing.values[:,0])):
            date1 = datetime.datetime.strptime(testing.values[i,0], '%Y-%m-%d').date()
            conv_dates = numpy.append(conv_dates, date1)
        plt.plot(conv_dates, testValue,'.', color='#ff3333',alpha=0.6)

        plt.xlabel('Date', fontsize=12, fontweight='bold', color='gray')
        plt.ylabel('Price', fontsize=12, fontweight='bold', color='gray')
        plt.show()
    ret = max(forecast.self_define_cycle)-min(forecast.self_define_cycle)
    model_tb = forecast['yhat']
    model_tb.index = forecast['ds'].map(lambda x:x.strftime("%Y-%m-%d"))
    

    Rpc = round(ret,3)
    temp = forecast['yhat']
    Mse = mean_squared_error(training["y"], temp[0:len(training["y"])] )
    Mae = mean_absolute_error(training["y"], temp[0:len(training["y"])] )
    
    MAX_rpc.checkBigger(Rpc, cycle)
    MIN_mse.checkSmaller(Mse, cycle)
    MIN_mae.checkSmaller(Mae, cycle)

    return 0

df = pd.read_csv('IBM_daily.csv', usecols=[0,4])
df.head()

for i in range(275,277):
    cycle_analysis(df, '2017-01-01', i, 'additive', forecast_plot=True)

MAX_rpc.printData("RPC maximum ")
MIN_mse.printData("MSE minimum ")
MIN_mae.printData("MAE minimum ")

print("Done.")

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