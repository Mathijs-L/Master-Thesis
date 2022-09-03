# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:46:47 2022

@author: mlenderi
"""
from math import sqrt
from numpy import split
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D
from keras import regularizers
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, Callback
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib import pyplot as plt




#%%
def lstm_data(df, timestamps):
    array = np.empty((0,df.shape[1]))
    range_ = df.shape[0]-(timestamps-1)
    for t in range(range_):
        dfp = df[t:t+timestamps, :]
        array = np.vstack((array, dfp))

    df_array = array.reshape(-1,timestamps, array.shape[1])
    #inverse_array = sc.inverse_transform(df_array)
    return df_array


    



#%%
df = pd.read_csv("AR_Items_All_CoCodes_2018_2022.csv", delimiter = ',',encoding = "UTF-16",header=0, infer_datetime_format=True, parse_dates=['Clearing_date'], index_col=['Clearing_date'])
df['Amount_gc_ecc'] = df['Amount_gc_ecc'].astype(int)


#summing amount per day/week
df = df.resample('W').sum()

#selection of dataset days
df = df["2018-01-01":"2020-05-01"]

#selection of amount variable
df = df.loc[:, ['Amount_gc_ecc']]
#df = pd.DataFrame(range(0,150))
#creating the lagged 1 year variable
df["Amount_lagged"] = df.shift(48)

#dropping data with no lag
#df = df.dropna(0)
df = df.values






#%% SPLITTING THE DATASET INTO TRAIN AND TEST 
dataset = df

#Number of validation and test weeks
N_val_weeks = 16
N_test_weeks = 32



#selecting only the outcome amount data
train_data = dataset[:len(dataset)-N_test_weeks-N_val_weeks]
val_data = dataset[len(dataset)-N_test_weeks-N_val_weeks:len(dataset)-N_test_weeks]
test_data = dataset[len(dataset)-N_test_weeks:]

scaler = MinMaxScaler()
scaled_train_data = scaler.fit_transform(train_data)
scaled_val_data = scaler.transform(val_data)
scaled_test_data = scaler.transform(test_data)

#combine the scaled datasets such that the data roll can be applied
dataset = np.vstack((scaled_train_data, scaled_val_data, scaled_test_data))

dataset = lstm_data(dataset, 4)





#splitting Y_df and X_df where X_df is rolled 4 weeks back such that the past 4 weeks forecast the upcoming 4 weeks
Y_df = dataset[:,:,:1]
X_df = np.roll(dataset, 4, axis = 0)


#Creating train set
Y_df_train = Y_df[:len(Y_df)-N_test_weeks-N_val_weeks,:,:]
X_df_train = X_df[:len(Y_df)-N_test_weeks-N_val_weeks,:,:]

#Creating validation set
Y_df_val = Y_df[len(Y_df)-N_test_weeks-N_val_weeks:len(Y_df)-N_test_weeks,:,:]
X_df_val = X_df[len(X_df)-N_test_weeks-N_val_weeks:len(X_df)-N_test_weeks,:,:]

#Creating test set
Y_df_test = Y_df[len(Y_df)-N_test_weeks:,:,:]
X_df_test = X_df[len(X_df)-N_test_weeks:,:,:]



#deleting first year from both due to the lag the shift created and the roll of 4 weeks (52+4)
#Only needed in train set because the X_df was shifted foreward
Y_df_train = Y_df_train[56:]
X_df_train = X_df_train[56:]


#%% LSTM MODEL         Best model till now 0.18696746359929242 is [480,128,32,32][164,1] batch size = 4, epoch= 100
#Defining training parameters 
verbose, epochs, batch_size = 1, 1000, 4

#Defining the model
model = Sequential()
model.add(LSTM(units = 32 ,return_sequences = False ,   activation='relu'))
model.add(Dropout(0))
model.add(Dense(96,  activation='relu'))
model.add(Dense(32,  activation='relu'))

model.add(Dense(4))
model.compile(loss='mse', optimizer='adam')

# fit network
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model.fit(X_df_train, Y_df_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data = (X_df_val, Y_df_val), callbacks = [es] )

#Prediction values from model
prediction = model.predict(X_df_test)

#Actual values
actual = Y_df_test[:,1]

actual_list = Y_df_test.tolist()



LSTM_actual_list = []
for i in actual_list:
    for p in i:
        LSTM_actual_list.append(p[0])
        
LSTM_prediction_list = []
for i in prediction:
    for p in i:
        LSTM_prediction_list.append(p)

LSTM_4_week_RMSE = mean_squared_error(LSTM_actual_list, LSTM_prediction_list, squared = False)
print(LSTM_4_week_RMSE)

from sklearn.metrics import mean_absolute_error
LSTM_4_week_MAE = mean_absolute_error(LSTM_actual_list, LSTM_prediction_list)
print(LSTM_4_week_MAE)

plt.plot(LSTM_actual_list, label = "actual")
plt.plot(LSTM_prediction_list, label = "prediction")
plt.legend()


#%%
#model.save("Com_LSTM_model")
#pd.DataFrame(LSTM_prediction_list).to_csv("Com_LSTM_pred.csv")
#pd.DataFrame(LSTM_actual_list).to_csv("Com_LSTM_actual_list.csv")




