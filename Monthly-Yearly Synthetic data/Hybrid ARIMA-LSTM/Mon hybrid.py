# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:54:37 2022

@author: mlenderi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:53:00 2022

@author: mlenderi
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

plt.rc("figure", figsize=(16,8))
plt.rc("font", size=14)

# First we'll simulate the synthetic data
def simulate_seasonal_term(periodicity, total_cycles, noise_std=1.,
                           harmonics=None):
    duration = periodicity * total_cycles
    assert duration == int(duration)
    duration = int(duration)
    harmonics = harmonics if harmonics else int(np.floor(periodicity / 2))

    lambda_p = 2 * np.pi / float(periodicity)

    gamma_jt = noise_std * np.random.randn((harmonics))
    gamma_star_jt = noise_std * np.random.randn((harmonics))

    total_timesteps = 100 * duration # Pad for burn in
    series = np.zeros(total_timesteps)
    for t in range(total_timesteps):
        gamma_jtp1 = np.zeros_like(gamma_jt)
        gamma_star_jtp1 = np.zeros_like(gamma_star_jt)
        for j in range(1, harmonics + 1):
            cos_j = np.cos(lambda_p * j)
            sin_j = np.sin(lambda_p * j)
            gamma_jtp1[j - 1] = (gamma_jt[j - 1] * cos_j
                                 + gamma_star_jt[j - 1] * sin_j
                                 + noise_std * np.random.randn())
            gamma_star_jtp1[j - 1] = (- gamma_jt[j - 1] * sin_j
                                      + gamma_star_jt[j - 1] * cos_j
                                      + noise_std * np.random.randn())
        series[t] = np.sum(gamma_jtp1)
        gamma_jt = gamma_jtp1
        gamma_star_jt = gamma_star_jtp1
    wanted_series = series[-duration:] # Discard burn in

    return wanted_series

duration = 52*5
periodicities = [4, 52]
num_harmonics = [1, 1]
std = np.array([1, 3])
np.random.seed(44444)

terms = []
for ix, _ in enumerate(periodicities):
    s = simulate_seasonal_term(
        periodicities[ix],
        duration / periodicities[ix],
        harmonics=num_harmonics[ix],
        noise_std=std[ix])
    terms.append(s)
terms.append(np.ones_like(terms[0]) * 10.)
series = pd.Series(np.sum(terms, axis=0))
df = pd.DataFrame(data={'total': series,
                        '10(3)': terms[0],
                        '100(2)': terms[1],
                        'level':terms[2]})
h1, = plt.plot(df['total'])
h2, = plt.plot(df['10(3)'])
h3, = plt.plot(df['100(2)'])
h4, = plt.plot(df['level'])
plt.legend(['Monthly+Yearly','Monthly','Yearly', 'level'])
plt.show()

#%%
y = df["total"]

#Selecting the number of test weeks
N_test_weeks = 32

#Splitting the dataset in train and validate sets
y_to_train= y[:len(y)-N_test_weeks]
y_to_test = y[len(y)-N_test_weeks:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
y_to_train_scaled = scaler.fit_transform(pd.DataFrame(y_to_train.values))
y_to_test_scaled = scaler.transform(pd.DataFrame(y_to_test))






#%%Stationairy test
from pmdarima.arima import ADFTest
adf_test = ADFTest(alpha = 0.05)
adf_test.should_diff(y_to_train_scaled)

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
#from pmdarima.arima import auto_arima as auto_ARIMA

#auto_ARIMA(y_to_train_scaled, m = 52, seasonal = True, trace = True)




#%%
from pmdarima.arima import ARIMA as pmdARIMA



#MANUAL PARAMETERS
arima_model = pmdARIMA((3,0,2),seasonal_order=(2, 0, 0, 52),trace=True)
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

#%% RMSE CALC
from sklearn.metrics import mean_squared_error

'''
RMSE_ARIMA = mean_squared_error(y_to_test_scaled, prediction,squared = False)
print(RMSE_ARIMA)

from sklearn.metrics import mean_absolute_error
MAE_ARIMA = mean_absolute_error(y_to_test_scaled, prediction)
print(MAE_ARIMA)
'''

#%%
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
residuals_arima = arima_model.resid()
residuals_arima = np.append(residuals_arima, residuals_test)
df = pd.DataFrame(residuals_arima)
#df = df.append(ARIMA_residuals)

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

#scaler = MinMaxScaler()
#scaled_train_data = scaler.fit_transform(train_data)
#scaled_val_data = scaler.transform(val_data)
#scaled_test_data = scaler.transform(test_data)

#combine the scaled datasets such that the data roll can be applied
dataset = np.vstack((train_data, val_data, test_data))

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
verbose, epochs, batch_size = 1, 1000, 16

#Defining the model
model = Sequential()
model.add(LSTM(units = 96 ,return_sequences = False ,   activation='relu'))
model.add(Dropout(0))
model.add(Dense(4))
model.compile(loss='mse', optimizer='adam')

# fit network
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model.fit(X_df_train, Y_df_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data = (X_df_val, Y_df_val), callbacks = [es] )

#Prediction values from model
prediction = model.predict(X_df_test)

#Actual values
actual_errors = Y_df_test

actual_errors_list = Y_df_test.tolist()

Error_calc_actual_list = []
for i in actual_errors_list:
    for p in i:
        for z in p:
            Error_calc_actual_list.append(z)
        
Error_calc_prediction_list = []
for i in prediction:
    for p in i:
        Error_calc_prediction_list.append(p)

LSTM_4_week_RMSE = mean_squared_error(Error_calc_actual_list, Error_calc_prediction_list, squared = False)
print(LSTM_4_week_RMSE)

plt.plot(Error_calc_actual_list, label = "actual")
plt.plot(Error_calc_prediction_list, label = "prediction")
plt.legend()
plt.show()



#%%------Read in ARIMA predictions
ARIMA_pred = read_csv("Mon_ARIMA_pred.csv", index_col = 0)

prediction_list = np.array(Error_calc_prediction_list)

ARIMA_pred = np.array(ARIMA_pred)
ARIMA_pred_list = []
for i in ARIMA_pred:
    for i in i:
        ARIMA_pred_list.append(i)

hybrid_prediction = ARIMA_pred_list + prediction_list[:len(ARIMA_pred_list)]


plt.legend()







#plt.plot(actual_list, label = "actual", linestyle = ":")
plt.plot(hybrid_prediction-actual_list, label = "hybrid")
plt.plot(np.array(ARIMA_pred_list)-np.array(actual_list), label = "ARIMA")
plt.axhline(y=0, color = "black",label = "level")
plt.legend()
plt.show()



from sklearn.metrics import mean_squared_error
RMSE_hybrid = mean_squared_error(actual_list, hybrid_prediction,squared = False)
print(RMSE_hybrid)

from sklearn.metrics import mean_absolute_error
MAE_hybrid = mean_absolute_error(actual_list, hybrid_prediction)
print(MAE_hybrid)

pd.DataFrame(hybrid_prediction).to_csv("Mon_Hybrid_pred.csv")



#%%
'''
residuals_test_array = np.array(residuals_test)
residuals_test_4_weeks = data_to_4_weeks(residuals_test_array, 32, 4)
residuals_test_4_weeks = np.array(residuals_test_4_weeks)

residuals_test_4_weeks_list = []
for i in residuals_test_4_weeks:
    for i in i:
        for i in i:
            residuals_test_4_weeks_list.append(i)


#plt.plot(prediction_list, label = "prediciton")
plt.plot(error_list, label = "LSTM actual")
plt.plot(actual_list-ARIMA_pred, label = "residuals")
plt.plot(residuals_test_4_weeks_list, label = "residuals test")
plt.xlim([100,112])
plt.legend()
'''