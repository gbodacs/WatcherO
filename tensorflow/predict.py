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
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

def create_dataset_predict(dataset, look_back=1):
	dataX = []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
	return numpy.array(dataX)

inputfile = sys.argv[1]

print ('Input file is "', inputfile, '"')

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe = read_csv(inputfile, usecols=[4])
dataset = dataframe.values
dataset = dataset.astype('float32')


test_size = len(dataset)
predictLen = 15
real_size = int(test_size/predictLen)*predictLen
dataset = dataset[test_size-real_size:,]



# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# load test sets
test = dataset[:,:]

# reshape into X=t and Y=t+1
look_back = 15
testX, testY = create_dataset_learn(test, look_back)

# reshape input to be [samples, time steps, features]
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# load model
print ('Load model...')
model = load_model(inputfile + "_model")
print ('Model loaded!')

# model.save(inputfile + "_model")

# make predictions
testPredict = model.predict(testX)

# testX = create_dataset_predict(test, look_back)
# testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# testPredict = model.predict(testX)

for x in range(predictLen):
	temp = testPredict[-1]
	test = numpy.append(test, [temp], axis=0 )
	testX = create_dataset_predict(test, look_back)
	testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
	testPredict = model.predict(testX)
	print( "Using previous: %.3f value predicted: %.3f" % (temp, testPredict[-1]) )

# invert predictions
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

truePredict = testPredict[-predictLen:]
testPredict = testPredict[0:-predictLen] #a masodik duplikalt erteket is visszuk, hogy folyamatos legyen a vonal

# calculate root mean squared error
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[look_back:, :] = testPredict

# true predictions for plotting
truePredictPlot = numpy.empty_like(dataset)
for x in range(predictLen):
	truePredictPlot = numpy.append(truePredictPlot, [[numpy.nan]], axis=0)
truePredictPlot[:, :] = numpy.nan
truePredictPlot[-predictLen:len(truePredictPlot), :] = truePredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(truePredictPlot)
plt.show()