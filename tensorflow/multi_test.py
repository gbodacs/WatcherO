import numpy
from numpy import array
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
inputModelFile = inputfile + "_multi_model"
print ('Input CSV file is '+inputfile)
print ('Input model file is '+inputModelFile)

#load and scale the input data
dataframe = read_csv(inputfile, usecols=[4])
dataSet = dataframe.values
dataSet = dataSet.astype('float32')
dataset_orig = dataSet

dataScaler = MinMaxScaler(feature_range=(0, 1))
dataSet = dataScaler.fit_transform(dataSet)
trainLen = int(len(dataSet)*.90)
testLen  = len(dataSet)-trainLen

dataSet, dataTestSet = dataSet[0:trainLen,:], dataSet[trainLen:trainLen+testLen,:]

#load and test volumen
volumeframe = read_csv(inputfile, usecols=[6])
volumeSet = volumeframe.values
volumeSet = volumeSet.astype('float32')
volumeScaler = MinMaxScaler(feature_range=(0, 1))
volumeSet = volumeScaler.fit_transform(volumeSet)

volumeSet, volumeTestSet = volumeSet[0:trainLen,:], volumeSet[trainLen:trainLen+testLen,:]

#Set learn train
# convert to [rows, columns] structure
dataSet = dataSet.reshape((len(dataSet), 1))
volumeSet = volumeSet.reshape((len(volumeSet), 1))

# train_size = int(len(dataset) * 0.9)
# test_size = len(dataset)-train_size
# train, test = dataset[0:train_size,:], dataset[train_size:train_size+test_size,:]
LearnSet = hstack((dataSet, volumeSet))
n_steps_in, n_steps_out = 10, 3
X, y = split_sequences(LearnSet, n_steps_in, n_steps_out)
n_features = X.shape[2]

TestSet = hstack((dataTestSet, volumeTestSet))
n_steps_in, n_steps_out = 10, 3
Xtest, ytest = split_sequences(TestSet, n_steps_in, n_steps_out)

# load model
print ('Load model...')
model = load_model(inputModelFile)
print ('Model loaded!')

# make predictions
trainPredict = model.predict(X)
testPredict = model.predict(Xtest)


# invert predictions
#trainPredict = scaler.inverse_transform(trainPredict)

# # calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))

# train predictions for plotting
trainPredictPlot1 = numpy.empty_like(dataSet)
trainPredictPlot1[:, :] = numpy.nan
trainPredictPlot1[n_steps_in-1:len(trainPredict)+n_steps_in-1, :] = trainPredict[:,0:1,0]

trainPredictPlot2 = numpy.empty_like(dataSet)
trainPredictPlot2[:, :] = numpy.nan
trainPredictPlot2[n_steps_in-1:len(trainPredict)+n_steps_in-1, :] = trainPredict[:,1:2,0]

trainPredictPlot3 = numpy.empty_like(dataSet)
trainPredictPlot3[:, :] = numpy.nan
trainPredictPlot3[n_steps_in-1:len(trainPredict)+n_steps_in-1, :] = trainPredict[:,2:3,0]

#
testPredictPlot1 = numpy.empty_like(dataset_orig)
testPredictPlot1[:, :] = numpy.nan
testPredictPlot1[len(trainPredict)+n_steps_in*2:len(trainPredict)+len(testPredict)+n_steps_in*2, :] = testPredict[:,0:1,0]

testPredictPlot2 = numpy.empty_like(dataset_orig)
testPredictPlot2[:, :] = numpy.nan
testPredictPlot2[len(trainPredict)+n_steps_in*2:len(trainPredict)+len(testPredict)+n_steps_in*2, :] = testPredict[:,1:2,0]

testPredictPlot3 = numpy.empty_like(dataset_orig)
testPredictPlot3[:, :] = numpy.nan
testPredictPlot3[len(trainPredict)+n_steps_in*2:len(trainPredict)+len(testPredict)+n_steps_in*2, :] = testPredict[:,2:3,0]


# plot baseline and predictions
plt.xlabel('Time (days)')
plt.ylabel('Price')
plt.suptitle("Data from: "+inputfile)
plt.plot(dataset_orig, color='#dddddd')
plt.plot(dataScaler.inverse_transform(trainPredictPlot1), color='#0533aa')
plt.plot(dataScaler.inverse_transform(trainPredictPlot2), color='#1144cc')
plt.plot(dataScaler.inverse_transform(trainPredictPlot3), color='#2255ee')

plt.plot(dataScaler.inverse_transform(testPredictPlot1), color='#5533aa')
plt.plot(dataScaler.inverse_transform(testPredictPlot2), color='#7744cc')
plt.plot(dataScaler.inverse_transform(testPredictPlot3), color='#9955ee')
plt.legend(('Original', 'Train next day', 'Train +2 day', 'Train +3 day', 'Test next day', 'Test +2 day', 'Test +3 day'),
           loc='upper left')
plt.show()