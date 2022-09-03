# -*- coding: utf-8 -*-
"""
Created on Wed May 18 08:30:53 2022

@author: mlenderi
"""

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


duration = 52 * 4
periodicities = [4, 52]
num_harmonics = [1, 2]
std = np.array([1, 3])
np.random.seed(8678309)

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
plt.legend(['total','monthly cycle','yearly cycle', 'level'])
plt.show()

#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt




#
#Selecting the column of interest or df of interest
#y = df['total']
#y = df['100(2)']
y = linear_data(224)


#Selecting the number of training weeks
N_train_weeks = 32

#Splitting the dataset in train and validate sets
y_to_train= y[:len(y)-N_train_weeks]
y_to_test = y[len(y)-N_train_weeks:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
y_to_train_scaled = scaler.fit_transform(pd.DataFrame(y_to_train.values))
y_to_test_scaled = scaler.transform(pd.DataFrame(y_to_test))



#data to 4 weeks function
def data_to_4_weeks(test, n_weeks, forecast_weeks):
    actual_4_week = []
    for i in range(n_weeks - forecast_weeks+1):
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

#%% SES function

import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing 

def forecast_loop(train, test, n_weeks, forecast_weeks):
    prediction_4_week = []
    for i in range(n_weeks - forecast_weeks + 1):
        train = np.append(train, test[i])
        
        fit = SimpleExpSmoothing(train).fit()
        prediction = fit.forecast(4)
        print(prediction)
        prediction_4_week.append(prediction)
        
    return prediction_4_week
prediction = forecast_loop(y_to_train_scaled, y_to_test_scaled, 32, 4)

prediction = np.array(prediction)

prediction_list = []
for i in prediction:
    for i in i:
        prediction_list.append(i)

from sklearn.metrics import mean_squared_error
RMSE_SES = mean_squared_error(actual_list, prediction_list,squared = False)
print(RMSE_SES)

from sklearn.metrics import mean_absolute_error
MAE_SES = mean_absolute_error(actual_list, prediction_list)
print(MAE_SES)

plt.plot(actual_list, label = "actual")
plt.plot(prediction_list, label = "prediction")
plt.legend()


#%%auto_arima
from pmdarima import auto_arima
arima_model = auto_arima(y_to_train_scaled, m=52, seasonal = True, trace = True)


#%%ARIMA Forecast

from pmdarima.arima import ARIMA as pmdARIMA

#forecast loop forecasts for n_weeks - forecast_weeks + 1 because it doesnt have to predict further thna the test data. this way in this case it predicts 4 weeks into the future. So the last prediction (with 32 training weeks) is week 29,30,31,32.

def forecast_loop(train, test, n_weeks, forecast_weeks):
    prediction_4_week = []
    for i in range(n_weeks - forecast_weeks + 1):
        train = np.append(train, test[i])
        
        arima_model = pmdARIMA((2,0,1),seasonal_order=(0, 1, 0, 52),trace=True)
        arima_model.fit(train)
        prediction = pd.DataFrame(arima_model.predict(n_periods = 4))
        print(prediction)
        prediction_4_week.append(prediction)
        
    return prediction_4_week
    
prediction = forecast_loop(y_to_train_scaled, y_to_test_scaled, 32, 4)

prediction = np.array(prediction)

prediction_list = []
for i in prediction:
    for i in i:
        for i in i:
            prediction_list.append(i)

from sklearn.metrics import mean_squared_error
RMSE_ARIMA = mean_squared_error(actual_list, prediction_list,squared = False)
print(RMSE_ARIMA)


from sklearn.metrics import mean_absolute_error
MAE_ARIMA = mean_absolute_error(actual_list, prediction_list)
print(MAE_ARIMA)

plt.plot(actual_list, label = "actual")
plt.plot(prediction_list, label = "prediction")
plt.legend()
