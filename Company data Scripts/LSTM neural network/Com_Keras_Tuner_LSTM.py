# -*- coding: utf-8 -*-
"""
Created on Wed May 18 08:03:09 2022

@author: mlenderi
"""

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
df = pd.read_csv("AllCoCo2015-2022.csv", delimiter = ',',encoding = "UTF-16",header=0, infer_datetime_format=True, parse_dates=['Clearing_date'], index_col=['Clearing_date'])
df['Amount_gc_ecc'] = df['Amount_gc_ecc'].astype(int)


#summing amount per day/week
df = df.resample('W').sum()

#selection of dataset days
df = df["2016-01-01":"2022-05-01"]

#selection of amount variable
df = df.loc[:, ['Amount_gc_ecc']]

#creating the lagged 1 year variable
df["Amount_lagged"] = df.shift(48)

#dropping data with no lag
#df = df.dropna(0)
df = df.values






#%% SPLITTING THE DATASET INTO TRAIN AND TEST 
dataset = df

#Number of validation and test weeks
N_val_weeks = 20
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
    for i in range(hp.Int('n_layers_LSTM', 0, 2)):  # adding variation of layers.
          model.add(LSTM(hp.Int(f'LSTM_{i}_units',
                                  min_value=32,
                                  max_value=128,
                                  step=32),activation = "relu" , return_sequences = True))
    model.add(LSTM(hp.Int('LSTM_out_units', min_value=32, max_value=128, step=32),activation='relu', return_sequences = False)) 
    for i in range(hp.Int('n_layers_Dense', 0, 2)):  # adding variation of layers.
          model.add(Dense(hp.Int(f'Dense_{i}_units',
                                    min_value=32,
                                    max_value=128,
                                    step=32),activation = "relu" ))
    model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.1)))
    #model.add(Dense(hp.Int('Dense_Layer_neurons',min_value=32,max_value=128,step=32), activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics = ['mse'])
    return model

es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
tuner = RandomSearch(build_model, objective='val_loss', max_trials=100,executions_per_trial=3, overwrite = True)
tuner.search(X_df_train, Y_df_train, batch_size = 4, epochs=200, callbacks = [es], validation_data=(X_df_val, Y_df_val))
best_model = tuner.get_best_models()[0]
best_parameters = tuner.get_best_hyperparameters(1)[0]
print(best_parameters.values)
