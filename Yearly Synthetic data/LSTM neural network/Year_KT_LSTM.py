
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




#%% LSTM DATA FUNCTION
def lstm_data(df, timestamps):
    array = np.empty((0,df.shape[1]))
    range_ = df.shape[0]-(timestamps-1)
    for t in range(range_):
        dfp = df[t:t+timestamps, :]
        array = np.vstack((array, dfp))

    df_array = array.reshape(-1,timestamps, array.shape[1])
    #inverse_array = sc.inverse_transform(df_array)
    return df_array


#%% LINEAR SYNTH DATA CREATOR
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt



def linear_data(n_weeks):
    series = range(0,n_weeks)
    return pd.DataFrame(series)


#%% CYCLIC SYNTH DATA CREATOR
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
plt.legend(['total','10(3)','100(2)', 'level'])
plt.show()


#%%LSTM

#Selecting the column of interest or df of interest
#y = df['total']
y = df['100(2)']


df = pd.DataFrame(y)
#df = linear_data(260)
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

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
tuner = RandomSearch(build_model, objective='val_loss', max_trials=30,executions_per_trial=1, overwrite = True)
tuner.search(X_df_train, Y_df_train, batch_size = 4, epochs=100, callbacks = [es], validation_data=(X_df_val, Y_df_val))
best_model = tuner.get_best_models()[0]
best_parameters = tuner.get_best_hyperparameters(1)[0]
print(best_parameters.values)
