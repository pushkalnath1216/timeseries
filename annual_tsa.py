import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

df=pd.read_csv('D://course docs//DATA SCIENCE//ML//datasets//annual_csv.csv',parse_dates=['Date'],index_col=['Date'])
df.head()

plt.xlabel('Date')
plt.ylabel('Price')
plt.plot(df)


rolling_mean=df.rolling(window=20).mean()
rolling_std=df.rolling(window=20).std()

plt.plot(df,color='blue',label='original')
plt.plot(rolling_mean,color='red',label='Rolling MEAN')
plt.plot(rolling_std,color='black',label='Rolling Std')
plt.legend(loc='best')
plt.title('ROLLING MEAN AND ST.DEVIATION')
plt.show()

result=adfuller(df['Price'])
ADF_statistic=result[0]
p_value=result[1]
print("critical_values:")
for key,value in result[4].items():
    print('\t',key,value)

df_log=np.log(df)
plt.plot(df_log)

def get_stationarity(timeseries):
    rolling_mean=timeseries.rolling(window=20).mean()
    rolling_std=timeseries.rolling(window=20).std()
    
    plt.plot(timeseries,color='blue',label='original')
    plt.plot(timeseries,color='red',label='rolling mean')
    plt.plot(timeseries,color='black',label='rolling std')
    plt.legend(loc='best')
    plt.title('rolling mean and st.deviation')
    plt.show(block=False)
    result=adfuller(timeseries['Price'])
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))
        
rolling_mean=df_log.rolling(window=20).mean()
df_log_minus_mean=df_log-rolling_mean   
df_log_minus_mean.dropna(inplace=True) 

get_stationarity(df_log_minus_mean)


rolling_mean_exp_decay=df_log.ewm(halflife=12,min_periods=0,adjust=True).mean()
df_log_exp_decay=df_log-rolling_mean_exp_decay
df_log_exp_decay.dropna(inplace=True)

get_stationarity(df_log_exp_decay)

df_log_shift=df_log-df_log.shift()
df_log_shift.dropna(inplace=True)
get_stationarity(df_log_shift)

#ARIMA:
decomposition=seasonal_decompose(df_log)
model=ARIMA(df_log,order=(2,1,2))
results=model.fit(disp=-1)
plt.plot(df_log_exp_decay)
plt.plot(results.fittedvalues,color='red')

predictions_arima_diff=pd.Series(results.fittedvalues,copy=True)
predictions_arima_diff_cumsum=predictions_arima_diff.cumsum()
predictions_arima_log=pd.Series(df_log['Price'].iloc[0],index=df_log.index)
predictions_arima_log=predictions_arima_log.add(predictions_arima_diff_cumsum,fill_value=0)
predictions_arima=np.exp(predictions_arima_log)
plt.plot(df)
plt.plot(predictions_arima)
results.plot_predict(1,264)


