# -*- coding: utf-8 -*-
"""
Created on Thu May 19 17:32:43 2022

@author: mlenderi
"""




#%%

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


#%%
#Loading in the dataset with the clearing date as index
#df = pd.read_csv("AR_Items_All_CoCodes_2018_2022.csv", delimiter = ',',encoding = "UTF-16",header=0, infer_datetime_format=True, parse_dates=['Clearing_date'], index_col=['Clearing_date'])
#Set the datatype of the amount column to type integer
#df['Amount_gc_ecc'] = df['Amount_gc_ecc'].astype(int)

#Sum the amount by week on the index of the clearing date
#df = df.resample('W').sum()

#Selecting the timespan of the dataset
#df = df["2016-01-01":"2022-05-01"]




#%%


def linear_data(n_weeks):
    series = range(0,n_weeks)
    return pd.DataFrame(series)

#%%
#Selecting the column of interest
y = linear_data(226)

#Selecting the number of test weeks
N_test_weeks = 32

#Splitting the dataset in train and validate sets
y_to_train= y[:len(y)-N_test_weeks]
y_to_test = y[len(y)-N_test_weeks:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
y_to_train_scaled = scaler.fit_transform(pd.DataFrame(y_to_train.values))
y_to_test_scaled = scaler.transform(pd.DataFrame(y_to_test))


#y_to_train = y[:'2022-02-01'] # dataset to train
#y_to_test = y['2022-02-01':]
plt.plot(y_to_train_scaled)
plt.plot(y_to_test_scaled)




#%%
def data_to_4_weeks(test, n_weeks, forecast_weeks):
    actual_4_week = []
    for i in range(n_weeks - forecast_weeks + 1):
        week = pd.DataFrame([test[i],test[i+1],test[i+2],test[i+3]])
        actual_4_week.append(week)
        
        
    return actual_4_week
    
actual = data_to_4_weeks(y_to_test_scaled, 32,4)
actual = np.array(actual)
actual_list = []
for i in actual:
    for i in i:
        for i in i:
            actual_list.append(i)

#%%
from pmdarima.arima import auto_arima as auto_ARIMA

#auto_ARIMA(y_to_train_scaled, m = 52, seasonal = True, trace = True)




#%%
from pmdarima.arima import ARIMA as pmdARIMA

#MANUAL PARAMETERS
arima_model = pmdARIMA((0,1,0),seasonal_order=(0, 0, 0, 52),trace=True)
arima_model.fit(y_to_train_scaled)
prediction = pd.DataFrame(arima_model.predict(n_periods = 32))

#calculate residuals of the test period
#ARIMA_residuals = pd.DataFrame(y_to_val[0] - prediction[0], index = y_to_val.index)

plt.figure(figsize = (8,5))
#plt.plot(y_to_train_scaled, label = "training")
plt.plot(y_to_test_scaled, label = "test")
plt.plot(prediction, label = "predicted")
plt.legend()

plt.show()

residuals_test = y_to_test_scaled - prediction




#%%

#pd.DataFrame(prediction_list).to_csv("Corrected_ARIMA_0_118740")





from math import sqrt
from numpy import split
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Dropout
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np




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
df = arima_model.resid()
df = np.append(df, residuals_test)
df = pd.DataFrame(df)

#creating the lagged 1 year variable
df["Amount_lagged"] = df.shift(48)

#dropping data with no lag
#df = df.dropna(0)
df = df.values






#%% SPLITTING THE DATASET INTO TRAIN AND TEST 
dataset = df

#Number of validation and test weeks
N_val_weeks = 32
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
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from keras.callbacks import EarlyStopping, Callback

def build_model(hp):
    model = Sequential()
    model.add(LSTM(hp.Int('input_unit',min_value=8,max_value=128,step=8),return_sequences=True))
    
    model.add(LSTM(hp.Int('layer_2_neurons',min_value=8,max_value=128,step=8),return_sequences = False))
    model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.8,step=0.1)))
    model.add(Dense(hp.Int('Dense_Layer_neurons',min_value=8,max_value=128,step=8), activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics = ['mse'])
    return model

es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
tuner = RandomSearch(build_model, objective='val_loss', max_trials=30,executions_per_trial=2, overwrite = True)
tuner.search(X_df_train, Y_df_train, batch_size = 4, epochs=100, callbacks = [es], validation_data=(X_df_val, Y_df_val))
best_model = tuner.get_best_models()[0]
best_parameters = tuner.get_best_hyperparameters(1)[0]
print(best_parameters.values)




#%%

