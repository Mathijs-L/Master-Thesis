# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 19:16:37 2022

@author: mlenderi
"""

#%%

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


#%%
#Loading in the dataset with the clearing date as index
df = pd.read_csv("AR_Items_All_CoCodes_2018_2022.csv", delimiter = ',',encoding = "UTF-16",header=0, infer_datetime_format=True, parse_dates=['Clearing_date'], index_col=['Clearing_date'])
#Set the datatype of the amount column to type integer
df['Amount_gc_ecc'] = df['Amount_gc_ecc'].astype(int)

#Sum the amount by week on the index of the clearing date
df = df.resample('W').sum()

#Selecting the timespan of the dataset
df = df["2018-01-01":"2020-05-01"]



#%%




#%%
#Selecting the column of interest
y = df['Amount_gc_ecc']

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



#forecast loop forecasts for n_weeks - forecast_weeks + 1 because it doesnt have to predict further thna the test data. this way in this case it predicts 4 weeks into the future. So the last prediction (with 32 training weeks) is week 29,30,31,32.

def forecast_loop(train, test, n_weeks, forecast_weeks):
    prediction_4_week = []
    for i in range(n_weeks - forecast_weeks + 1):
        
        
        arima_model = pmdARIMA((5,1,2),seasonal_order=(2, 0, 0, 52),trace=True)
        arima_model.fit(train)
        prediction = pd.DataFrame(arima_model.predict(n_periods = forecast_weeks))
        train = np.append(train, test[i])
        print(prediction)
        prediction_4_week.append(prediction)
        
    return prediction_4_week
    
forecast_weeks = 4

prediction = forecast_loop(y_to_train_scaled, y_to_test_scaled, N_test_weeks, forecast_weeks)

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
plt.plot(prediction_list, label = "ARIMA prediction")
plt.legend()

#%%

pd.DataFrame(prediction_list).to_csv("Com_ARIMA_pred.csv")





