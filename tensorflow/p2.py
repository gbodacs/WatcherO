
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
from matplotlib import pyplot

import numpy
from numpy import array
from numpy import hstack
import matplotlib.pyplot as plt
from pandas import read_csv
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
        plt.show()
    ret = max(forecast.self_define_cycle)-min(forecast.self_define_cycle)
    model_tb = forecast['yhat']
    model_tb.index = forecast['ds'].map(lambda x:x.strftime("%Y-%m-%d"))
    

    Rpc = round(ret,3)
    Mse = mean_squared_error(training["y"], forecast['yhat'])
    Mae = mean_absolute_error(training["y"], forecast['yhat'])        
    
    MAX_rpc.checkBigger(Rpc, cycle)
    MIN_mse.checkSmaller(Mse, cycle)
    MIN_mae.checkSmaller(Mae, cycle)

    return 0

df = pd.read_csv('VIX_daily.csv', usecols=[0,4])
df.head()

for i in range(15,320):
    cycle_analysis(df, '2017-01-01', i, 'additive', forecast_plot=False)

MAX_rpc.printData("RPC maximum ")
MIN_mse.printData("MSE minimum ")
MIN_mae.printData("MAE minimum ")

print("Done.")