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
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

#load and test volumen
volumeframe = read_csv(inputfile, usecols=[6])
volumeset = volumeframe.values
volumeset = volumeset.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
volumeset = scaler.fit_transform(volumeset)

#Set learn train
out_seq = array([dataset[i]*2 for i in range(len(volumeset))])
# convert to [rows, columns] structure
dataset = dataset.reshape((len(dataset), 1))
volumeset = volumeset.reshape((len(volumeset), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

dataset = hstack((dataset, volumeset))
n_steps_in, n_steps_out = 10, 3
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
n_features = X.shape[2]

# define model
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=2)

# save model
model.save(inputfile+"_multi_model")

# demonstrate prediction
x_input = array([[60, 65], [70, 75], [80, 85]])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)