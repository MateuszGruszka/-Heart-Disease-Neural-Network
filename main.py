from keras.models import Sequential
from keras.layers import Dense
import numpy as np
# fix random seed for reproducibility
np.random.seed(7)
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import optimizers

#read data
data = pd.read_csv("./data.csv")

#delete rows with missing data
data = data[data.ca != '?']
data = data[data.thal != '?']

#split data to training and test set (80-20)
X_train_df, X_test_df = train_test_split(data, test_size=0.2, shuffle=True)

#creating output classes
Y_train_df = X_train_df['num']
Y_test_df = X_test_df['num']

#deleting output classes from training data
X_train_df = X_train_df.drop(['num'], axis = 1)
X_test_df = X_test_df.drop(['num'], axis = 1)

#converting from datafram to numpy array
X_train = X_train_df.values
X_test = X_test_df.values
Y_train = Y_train_df.values
Y_test = Y_test_df.values

#changing values from 1,2,3,4 of illness to 1
Y_train[Y_train > 0] = 1
Y_test[Y_test > 0] = 1

#casting values to float
X_train = X_train.astype('float')
X_test = X_test.astype('float')
Y_train = Y_train.astype('float')
Y_test = Y_test.astype('float')

# create model
model = Sequential()

#TEST 1
# model.add(Dense(10, kernel_initializer='uniform', input_dim=13, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# sgd = optimizers.SGD(lr=0.01)
# model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

#TEST 2
# model.add(Dense(14, kernel_initializer='uniform', input_dim=13, activation='relu'))
# model.add(Dense(12, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# sgd = optimizers.SGD(lr=0.02)
# model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

#TEST 3
# model.add(Dense(13, kernel_initializer='uniform', input_dim=13, activation='relu'))
# model.add(Dense(5, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# sgd = optimizers.SGD(lr=0.7)
# model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

#TEST 4
# model.add(Dense(8, input_dim=13, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# sgd = optimizers.SGD(lr=0.001)
# model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

#TEST 5
# model.add(Dense(13, input_dim=13, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# sgd = optimizers.SGD(lr=0.001)
# model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

#TEST 6
# model.add(Dense(18, input_dim=13, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(14, activation='relu'))
# model.add(Dense(12, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(6, activation='relu'))
# model.add(Dense(6, activation='relu'))
# model.add(Dense(6, activation='relu'))
# model.add(Dense(6, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# sgd = optimizers.SGD(lr=0.001)
# model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

#TEST 7
# model.add(Dense(13, input_dim=13, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# adam = optimizers.Adam(lr=0.001)
# model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

#TEST 8
# model.add(Dense(13, input_dim=13, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# adam = optimizers.Adam(lr=0.0001)
# model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])


#TEST 9
# model.add(Dense(13, input_dim=13, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# adam = optimizers.Adam(lr=0.001)
# model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

#TEST 10

#layers of the model
# relu -> activation(x) = max(0,x)
model.add(Dense(13, input_dim=13, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#optymalizer settings, in this examlpe we changed default learning rate
adam = optimizers.Adam(lr=0.002)

#configurate model and picking optymalizer, loss function and metrics.
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

# Start learning
# batch_size is number of samples per gradient update
# epoch is an iteration over the entire data provided
#fit - train data

model.fit(X_train, Y_train, epochs=150, batch_size=10)


# evaluate the model
# acc = number of correct pred / total num of pred
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))