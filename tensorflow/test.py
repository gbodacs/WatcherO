import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import random
import sys, getopt
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#
# learn.py: learn the whole database and save the model
# test.py: test the model with the learned database
#	- use some params to predict not only one element in the future
# predict.py: predict the future using the saved model and a database
#
# convert an array of values into a dataset matrix
def create_dataset_learn(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		dataX.append([dataset[i, 0]])
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

def create_dataset_predict(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)):
		dataX.append([dataset[i, 0]])
	return numpy.array(dataX)

inputfile = sys.argv[1]

print ('Input file is "', inputfile, '"')

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
#url = "IBM_monthly.csv"
dataframe = read_csv(inputfile, usecols=[4])
dataframe = dataframe[0:288]
epochNumber = 100
train_size = int(len(dataframe) * 0.9)
test_size = len(dataframe)-train_size
predictLen = 10

dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# load test sets
train, test = dataset[0:train_size,:], dataset[train_size:train_size+test_size,:]

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset_learn(train, look_back)
testX, testY = create_dataset_learn(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# load model
print ('Load model...')
model = load_model(inputfile + "_model")
print ('Model loaded!')

# # create and fit the LSTM network
# model = Sequential()
# model.add(LSTM(4, input_shape=(1, look_back)))
# model.add(Dropout(0.2))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(trainX, trainY, epochs=epochNumber, batch_size=1, verbose=2)
# model_lstm.add(LSTM(256, dropout = 0.3, recurrent_dropout = 0.3))
# model_lstm.add(Dense(256, activation = 'relu'))
# model_lstm.add(Dropout(0.3))
# model_lstm.add(Dense(5, activation = 'softmax'))

# model.save(inputfile + "_model")

# make predictions
trainPredict = model.predict(trainX)
#testPredict = model.predict(testX)

testX = create_dataset_predict(test, look_back)
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
testPredict = model.predict(testX)

for x in range(predictLen):
	temp = testPredict[-1]
	test = numpy.append(test, [temp], axis=0 )
	testX = create_dataset_predict(test, look_back)
	testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
	testPredict = model.predict(testX)
	print( "Using previous: %.3f value predicted: %.3f" % (temp, testPredict[-1]) )

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

truePredict = testPredict[-predictLen:]
testPredict = testPredict[0:-predictLen]

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:-2,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift test predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[0:len(trainPredict), :] = trainPredict

testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[-len(testPredict):, :] = testPredict

truePredictPlot = numpy.empty_like(dataset)
for x in range(predictLen):
	truePredictPlot = numpy.append(truePredictPlot, [[numpy.nan]], axis=0)
truePredictPlot[:, :] = numpy.nan
truePredictPlot[-predictLen:len(truePredictPlot), :] = truePredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.plot(truePredictPlot)
plt.show()