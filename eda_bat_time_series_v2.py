# this is for testing git
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from math import sqrt

# Importing the dataset
df_prc = pd.read_csv('Winfield_prices.csv')
df_qty = pd.read_csv('Winfield_quantities.csv')

# Start working on the products one by one. 
product = 10
prc_list =  list(df_prc)
ind = prc_list[product]
rolmean_window = 12

# preparing the data
wdf = pd.DataFrame()
wdf['date'] = df_qty.Date
wdf['price'] = df_prc[ind]
wdf['qty'] = df_qty[ind]
wdf['date'] = pd.to_datetime(wdf['date'], dayfirst=True)
wdf = wdf[(wdf.qty != 0)]
wdf.set_index(['date'], inplace=True)
wdf.index.name=None


# Visualizing the initial data
#wdf.qty.plot(figsize=(12,8), title= 'Weekly Sales of %s' %ind, fontsize=14)
#plt.savefig('weekly_ridership_%s.png' %ind, bbox_inches='tight')

# Apply decomposition
#decomposition = seasonal_decompose(wdf.qty, freq=12)  
#fig = plt.figure()  
#fig = decomposition.plot()  
#fig.set_size_inches(15, 8)

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=rolmean_window,center=False).mean()
    rolstd = timeseries.rolling(window=rolmean_window,center=False).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    
test_stationarity(wdf.qty)
# P-value = 0.9 : not stationary

# check the stationarity of the log of data
#wdf.qty_log= wdf.qty.apply(lambda x: np.log(x))  
#test_stationarity(wdf.qty_log)
# P-value = 0.95 : not stationary

# apply differencing 
wdf['first_difference'] = wdf.qty - wdf.qty.shift(1)  
test_stationarity(wdf.first_difference.dropna(inplace=False))
# P-value = 3e-12 : stationary

# apply the log of first differencing
#wdf['log_first_difference'] = wdf.qty_log - wdf.qty_log.shift(1)  
#test_stationarity(wdf.log_first_difference.dropna(inplace=False))
# P-value = 2.5e-14 : stationary

# seasonal differencing (4 months)
#wdf['seasonal_difference'] = wdf.qty - wdf.qty.shift(4)
#test_stationarity(wdf.seasonal_difference.dropna(inplace=False))
# P-value = 0.27 : not stationary

#wdf['log_seasonal_difference'] = wdf.qty_log - wdf.qty_log.shift(28)  
#test_stationarity(wdf.log_seasonal_difference.dropna(inplace=False))
# P-value = 0.17 : not stationary

#wdf['seasonal_first_difference'] = wdf.first_difference - wdf.first_difference.shift(4)  
#test_stationarity(wdf.seasonal_first_difference.dropna(inplace=False))
# P-value = 3.5e-23 : stationary

#wdf['log_seasonal_first_difference'] = wdf.log_first_difference - wdf.log_first_difference.shift(28)  
#test_stationarity(wdf.log_seasonal_first_difference.dropna(inplace=False))
# P-value = 1.9e-24 : stationary

#fig = plt.figure(figsize=(12,8))
#ax1 = fig.add_subplot(211)
#fig = sm.graphics.tsa.plot_acf(wdf.first_difference.iloc[1:], lags=60, ax=ax1)
#ax2 = fig.add_subplot(212)
#fig = sm.graphics.tsa.plot_pacf(wdf.first_difference.iloc[1:], lags=60, ax=ax2)

#mod = sm.tsa.statespace.SARIMAX(wdf.qty, trend='n', order=(0,1,0), seasonal_order=(0,1,1,4))
#results = mod.fit()
#print (results.summary())

mod = sm.tsa.statespace.SARIMAX(wdf.qty, order=(1,1,0))
results = mod.fit()
print (results.summary())

wdf['forecast'] = results.predict(start = len(wdf.qty)-4, end= len(wdf.qty), dynamic= True)  
wdf[['qty', 'forecast']].plot(figsize=(12, 8)) 
plt.savefig('ts_predict_%s.png' %ind, bbox_inches='tight')

rms = np.mean((wdf.forecast[-4:] - wdf.qty[-4:])/wdf.qty[-4:]*100)
print(rms)    
