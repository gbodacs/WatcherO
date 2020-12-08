# LSTM for international airline passengers problem with regression framing
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility

#arr = numpy.array([1, 2, 3, 4, 5])

dataset = numpy.array([[10],[20],[30],[40],[50],[60],[70],[80],[90],[100],[110],[120],[130]])

bla1, bla2 = create_dataset(dataset, 3)

print("----------")
print(bla1)
print("----------")
print(bla2)
print("----------")