
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

def cycle_analysis(data, split_len, cycle,mode='additive', forecast_plot = False, print_ind=False):
    trainLen = int(len(data)*.90)
    testLen  = len(data)-trainLen
    training, testing = data[0:trainLen,:], data[trainLen:trainLen+testLen,:]
    
    predict_period = len(pd.date_range(trainLen, max(data.index)))
    df = training.reset_index()
    df.columns = ['ds','y']
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
    out_tb = pd.concat([testing,model_tb],axis=1)
    out_tb = out_tb[~out_tb.iloc[:,0].isnull()]
    out_tb = out_tb[~out_tb.iloc[:,1].isnull()]
    mse = mean_squared_error(out_tb.iloc[:,0],out_tb.iloc[:,1])
    rep = [ret,mse]
    if print_ind:
        print ("Projected return per cycle: {}".format(round(rep[0],2)))
        print ("MSE: {}".format(round(rep[1],4)))
    return rep

df = pd.read_csv('VIX_daily.csv', usecols=[0,4])
df.head()

for i in range(320):
    cycle_analysis(df, 0.9, i+50, 'additive', forecast_plot=False, print_ind=False)

print("Done.")