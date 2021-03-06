import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools

from pandas import Series
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from statsmodels.graphics.tsaplots import plot_pacf

timeSeriesName  = Series.from_csv('file_path.csv',header=0)
arimaOrder = #(p,d,q) parameters defined by the user, might be defined using an "ARIMA(p,d,q)_parameters" function to choose the parameters with the lowest MSE

n = 0.66 

#Use a length "n" corresponding to the length of your base, this value must be the amount of time you want to see in the past, being less than 1 and greater than 0. 
#The amount of points foreseen in future periods will be 1- n. 


#p = AR, which means 'autorregressive'
#i = I, which means 'integrated'
#q = MA, which means 'moving average'

timeSeriesName.describe()

timeSeriesName.plot()
pyplot.show()

#data visualization to check for seasonalities

plt.plot(timeSeriesName.groupby(pd.Grouper(freq='M')).mean())
plt.xlabel('date')
plt.ylabel('qty')
plt.xticks(rotation=45)
plt.title('Monthly average')
plt.show()

plt.plot(timeSeriesName.groupby(pd.Grouper(freq='B')).mean())
plt.xlabel('date')
plt.ylabel('qty')
plt.xticks(rotation=45)
plt.title('Bimonthly average')
plt.show()

plt.plot(timeSeriesName.groupby(pd.Grouper(freq='Q')).mean())
plt.xlabel('date')
plt.ylabel('qty')
plt.xticks(rotation=45)
plt.title('Quarterly average')
plt.show()

decomposition = seasonal_decompose(timeSeriesName.groupby(pd.Grouper(freq='M')).mean(),model='multiplicative')
decomposition.plot()

X = timeSeriesName.values

#Visualization of autocorrelation function 'acf' and partial autocorrelation function 'pacf'. 
#You must use 'acf' to determine 'MA' and 'PACF' to determine 'AR'. On this situation, there are normally three different scenarios:
#First: A large peak in the first lag, which decreases after a few lags, means you have a 'MA' term on your data. 
#Second: A large peak in the first lag, followed by a minimized wave that alternates between positive and negative correlations. That means you have a 'MA'>1.
#Third: Significant correlations in the first or second lag, followed by correlations which are not significant. That means you have an 'AR' term on your data.

plot_pacf(X, lags=100)
pyplot.show()

autocorrelation_plot(X)

#Check the stationarity of your time series using an Augmented Dickey Fuller test. 
#For a result greater than your critical value, you must differentiate your time series until your series get stationary in order to do any predictions.

result = adfuller(X)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
    
x = timeSeriesName.values
train_size = int(len(x) * n)

train, val = x[:train_size], x[train_size:]

#Prediction Train
model = ARIMA(timeSeriesName, order=arimaOrder)
model_fit = model.fit(disp=0)
print(model_fit.summary())

residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.title('Residual Error')
pyplot.show()
    
residuals.plot(kind='kde')
plt.xlabel('Residual')
pyplot.show()

print(residuals.describe())

history = [x for x in train]
prediction_val = list()
for t in range(len(val)):
    model = ARIMA(history,order=(arimaOrder))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    prediction_val.append(int(output[0]))
    history.append(val[t])
 
error = mean_squared_error(prediction_val,val)
print('Mean Squared Error = {}'.format(error))

plt.plot(prediction_val,color='red')
plt.xlabel('date')
plt.ylabel('timeSeriesName')
plt.title('Prediction on Validation set')
plt.show()

pyplot.plot(val)
pyplot.plot(prediction_val, color='red')
pyplot.show()

#Prediction for future values
history = [x for x in timeSeriesName]
prediction_test = list()
for t in range(k):  #k is an integer defined by the user, which is lesser or equal than '(1 - n)*length(x)'
    model = ARIMA(history,order=(arimaOrder))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    prediction_test.append(int(output[0]))
    history.append(val[t])

dates = pd.date_range('Day/Month/Year',periods=k,freq='D or W or M or Y')
prediction_test_df = pd.DataFrame(prediction_test,index=dates)
prediction_test_df.columns = ['timeSeriesName']

plt.plot(prediction_test_df,color='green')
plt.xlabel('date')
plt.ylabel('timeSeriesName')
plt.xticks(rotation=45)
plt.show()

submission = pd.DataFrame({'timeSeriesName':prediction_test})

submission.to_csv('filePath/result.csv',header=0)

complete_submission = pd.DataFrame({'qty':[]})
complete_submission = complete_submission.append(submission)

complete_submission.to_csv('submission.csv',index=False)
