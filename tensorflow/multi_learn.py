# multivariate multi-step encoder-decoder lstm example
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


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

inputfile = sys.argv[1] 
outputFile = inputfile + "_multi_" + "_model"
print ('Input CSV file is '+inputfile)
print ('Output model file is '+outputFile)

#load and scale the input data
dataframe = read_csv(inputfile, usecols=[4])
dataset = dataframe.values
dataset = dataset.astype('float32')
scalerData = MinMaxScaler(feature_range=(0, 1))
dataset = scalerData.fit_transform(dataset)
trainLen = int(len(dataset)*0.9)
dataset= dataset[0:trainLen,:] # levagjuk a tesztadatokat a vegerol

#load and test volumen
volumeframe = read_csv(inputfile, usecols=[6])
volumeset = volumeframe.values
volumeset = volumeset.astype('float32')
scalerVol = MinMaxScaler(feature_range=(0, 1))
volumeset = scalerVol.fit_transform(volumeset)
volumeset= volumeset[0:trainLen,:] # levagjuk a tesztadatokat a vegerol

#Set learn train

# convert to [rows, columns] structure
dataset = dataset.reshape((len(dataset), 1))
volumeset = volumeset.reshape((len(volumeset), 1))

MixDataSet = hstack((dataset, volumeset))
n_steps_in, n_steps_out = 4, 2
X, ytemp = split_sequences(MixDataSet, n_steps_in, n_steps_out)
n_features_in = X.shape[2]
n_features_out = 1
Xtemp, y = split_sequences(dataset, n_steps_in, n_steps_out)

# define model
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features_in)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features_out, activation="sigmoid")))
model.compile(loss='categorical_crossentropy', optimizer="RMSprop", metrics=['acc'])
model.fit(X, y, epochs=100, verbose=2, )

# save model
model.save(inputfile+"_multi_model")
