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
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

inputfile = sys.argv[1]

print ('Input file is "', inputfile, '"')

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
#url = "IBM_monthly.csv"
dataframe = read_csv(inputfile, usecols=[4])
epochNumber = 2
train_size = int(len(dataframe) * 0.9)
test_size = len(dataframe)-train_size
predictLen = 25

dataset = dataframe.values
#dataset = dataset[0:train_size+test_size+predictLen]
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# load test sets
train, test = dataset[0:train_size,:], dataset[train_size:train_size+test_size,:]

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# load model
#model = load_model(inputfile)
# # create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=epochNumber, batch_size=1, verbose=2)

#model.save(inputfile + "_model")

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

for x in range(predictLen):
    temp = testPredict[len(testPredict)-1]
    print('Added predict: %.2f value' % (temp))
    test = numpy.append(test, [temp], axis=0 )
    testX, testY = create_dataset(test, look_back)
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

truePredict = testPredict[-predictLen:]
testPredict = testPredict[0:-predictLen]

# shift test predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[-len(testPredict):, :] = testPredict

truePredictPlot = numpy.empty_like(dataset)
for x in range(predictLen):
	truePredictPlot = numpy.append(truePredictPlot, [[numpy.nan]], axis=0)
truePredictPlot[:, :] = numpy.nan
truePredictPlot[-predictLen-1:len(truePredictPlot)-1, :] = truePredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.plot(truePredictPlot)
plt.show()