Accounts Receivable Cash Flow Forecasting: 
A comparison between SES, ARIMA, LSTM and Hybrid ARIMA-LST
Author: Mathijs Lenderink
Supervisor: dr. Gertjan Verhoeven
Date: 3-6-2022
Grade: 8


This repository contains all python scripts I used to program the four models I analyze in my master thesis.

For a full explanation of the method and models I use you can read the pdf of my master thesis which is also included in this repository.


A short summary of my master thesis project:
I analyzed four different models on how well they performed forecasting cashflow data of the accounts receivable. 
The data I used consisted of a summed amount of all the cashflow per week. 
To check the models I also used three synthetic datasets which all had a different pattern in the data.
The first synthetic dataset was a dataset with a linear pattern.
The second synthetic dataset had a yearly repeating pattern.
The third synthetic data had a yearly combined with a monthly repeating pattern.
The graphs showing these patterns can be found in the pdf file of my master thesis.
All the models were optimized using their own optimizers. 
This is also why there is a seperate file for the Long Short-Term Memoery neural network optimization as I used the Keras Tuner to optimize those models.
Each model was made to forecast a four week window and to analyze which models performed best these forecasts were compared using the Root Mean Squared Error and the Mean Absolute Error.


The models I used were the following:
Simple Exponential Smoothing
Seasonal ARIMA (Autoregressive integrated moving average)
Long Short-Term Memory neural network
Hybrid model consisting of seasonal ARIMA combined with a Long SHort-Term Memory neural network



Here is the abstract of my master thesis:
Forecasting cash flow is important to ensure the health of a company. This paper 
will research to what extent machine learning techniques can predict the cash 
flow of the accounts receivable based on historical data at a company of which 
the accounts receivable consists of few large value entries and many small value 
entries. In this study I compare the performance of SES, ARIMA, LSTM and 
Hybrid ARIMA-LSTM models on one company dataset and three synthetic 
datasets. The models forecast a four-week window and the model performance 
is analyzed using the root mean squared error and mean absolute error. The 
results of the synthetic datasets showed that the hybrid model gave the most 
accurate prediction. The results of the company dataset show that for this 
company there was not one best model. Further research is needed to conclude 
whether this is also the case for similar other companies. 



Keywords: Time series forecasting, cash flow, Accounts receivable (AR), Simple 
Exponential Smoothing (SES), Autoregressive Integrated Moving Average 
(ARIMA), Long Short-Term Memory neural network (LSTM), Hybrid ARIMALSTM, Synthetic data, Auto_arima, Keras Tuner, Root mean squared error 
(RMSE), Mean absolute error (MAE) 
