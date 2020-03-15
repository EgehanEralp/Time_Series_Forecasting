#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 14:12:19 2019

@author: egehaneralp
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from datetime import datetime
from statsmodels.tsa.stattools import adfuller,acf,pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.impute import SimpleImputer

from statsmodels.tsa.arima_model import ARIMA

#plot grafiklerinin styling ayarı
rcParams['figure.figsize'] = 10, 6

veriler = pd.read_csv('2017finansıXU100.csv')

#x ekseninde kullanılacak olan ZAMAN(t) parametresni Date&Tıme a çevirdi

veriler2 = veriler.drop(['Open','High','Low','Adj Close','Volume'],axis=1)
#index sütunu <- zaman sütunu (x ekseni olarak kullanacağımız)
veriler['Date'] = pd.to_datetime(veriler['Date'], infer_datetime_format=True)

veriler2 = veriler2.set_index(['Date'])

satirsayi=len(veriler2.index)

imputerNum = SimpleImputer(missing_values=np.nan,strategy="mean")
numerikveriler = veriler2.select_dtypes(include=['int64','float64'])
listX=list(numerikveriler.columns.values)
imputerNum = imputerNum.fit(numerikveriler)#HER KOLON  için ayrı ayrı  ortalama değer işlemi UYGULAR
numerikveriler = imputerNum.transform(numerikveriler) #float döndürür

#float64 to DataFrame yaotım.
veriler2 = pd.DataFrame(data=numerikveriler, index = range(satirsayi), columns=listX)


#%%
""" --------- CURRENT STATE GRAPH ------ """
plt.xlabel('Time')
plt.ylabel('Number of Passengers')
plt.plot(veriler2)


""" FORECAST YAPACAĞIN GRAFİK İÇİN :
    STABİL OLMALI (STATIONARY)-> MEAN VE STD CONSTANT GİBİ DAVRANMALI (PREPROCESSİNG İŞLEMİ)
"""
#%% 
"""  GRAFİĞE ENTEGRE EDİLECEK DEVAMLI MEAN VE STD SAPMA DEĞERLERİNİN TESPİTİ  """

rollingMean = veriler2.rolling(window=12).mean()   #window -> 365 için günlük analiz, 12 için aylık analiz yapar
rollingStdDev = veriler2.rolling(window=12).std()

original = plt.plot(veriler2,color='black',label='Orjinal')
meanPlt = plt.plot(rollingMean, color='blue',label='Ortalama')
stddevPlt=plt.plot(rollingStdDev, color='red',label='Standart Sapma')

plt.legend(loc='best')
plt.title('Ortalama & Standart Sapma')
plt.show(block=False)

#%% ERROR
"""  PERFORM DICKEY-FULLER TEST  """
dftest = adfuller(veriler2['Close'],autolag='AIC')

dfoutput=pd.Series(dftest[0:4],index=['Test stats','p-value','#Lags Used','Num of Obs'])
for key,value in dftest[4].items():
    dfoutput['Crit Value (%s)' %key] =value
print(dfoutput)
# sonuca göre p-value 0.5 civarlarında olmalıdır -> 0.99 => Data is not stationary
# => Estimate the train with LOG
# p-value değerine göre SABİT veriler olup olmadığına bakarız.

#%%
"""  Veriler Stationary(stabil) değil ise STABİL hale getirilir  """
"""  Bunun için LOGARİTMA  kullanılır  """
"""  Büyük sayılarca yükselen verilerde logaritma ile mevcut yükselik düşük sayılara göre SCALE edilir.  """
#NP.LOG -> e^n olarak logaritma alır.
veriler2_log = np.log(veriler2)
plt.plot(veriler2_log)
#%%
#LOG grafiği çizdirmek
movingAverage = veriler2_log.rolling(window=12).mean()
movingStd = veriler2_log.rolling(window=12).std()
plt.plot(veriler2_log)
plt.plot(movingAverage, color='red')

#MEAN ISNT STATIONARY =>
#%%
veriler2LogMinusMovAv = veriler2_log - movingAverage #standart sapma için
veriler2LogMinusMovAv.head(12)

#NaN değerlerini kaldırmak !!!
veriler2LogMinusMovAv.dropna(inplace=True)
veriler2LogMinusMovAv.head(10)
#%%
""" FULL FUNCTION OF STATIONARY TESTING OF GRAPHIC """
from statsmodels.tsa.stattools import adfuller
def test_stationary(timeseries):
    movingAverage = timeseries.rolling(window=12).mean()   #window -> 365 için günlük analiz, 12 için aylık analiz yapar
    movingStd = timeseries.rolling(window=12).std()
    
    original = plt.plot(timeseries,color='blue',label='Orjinal')
    meanPlt = plt.plot(movingAverage, color='red',label='Ortalama')
    stddevPlt=plt.plot(movingStd, color='black',label='Standart Sapma')    
    plt.legend(loc='best')
    plt.title('Ortalama & Standart Sapma')
    plt.show(block=False)
    
    print(':::: Dicky-Fuller test sonuclarım ::::')
    dftest = adfuller(timeseries['Close'],autolag='AIC')
    dfoutput=pd.Series(dftest[0:4],index=['Test stats','p-value','#Lags Used','Num of Obs'])
    for key,value in dftest[4].items():
        dfoutput['Crit Value (%s)' %key] =value
    print(dfoutput)
#%%
test_stationary(veriler2LogMinusMovAv)

#%%    TRANSFORMATIONS
#1
expDecayWeightedAvg = veriler2_log.ewm(halflife=12, min_periods=0, adjust=True).mean()
plt.plot(veriler2_log)
plt.plot(expDecayWeightedAvg,color='red')
#%% ERROR
#2
veriler2LogMinusExpDWA = veriler2_log - expDecayWeightedAvg
test_stationary(veriler2LogMinusExpDWA)
"""  şuanda stabil bir standart sapma ve ortalama elde etmiş olduk.  """
"""  P-VALUE==0.005  """
#%%
shiftedv2=veriler2_log.shift()
veriler2logShifting = veriler2_log - shiftedv2
plt.plot(veriler2logShifting)

veriler2logShifting.dropna(inplace=True)
test_stationary(veriler2logShifting)

#%% ERROR
decomposition = seasonal_decompose(veriler2_log)

trend=decomposition.trend
seasonal=decomposition.seasonal
residual=decomposition.resid

plt.subplot(411)
plt.plot(veriler2_log,label="ORGINAL")
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label="Trend")
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label="Seasonal")
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label="Residual")
plt.legend(loc='best')
plt.tight_layout()

decomposedLogData = residual
decomposedLogData.dropna(inplace=True)
test_stationary(decomposedLogData)
#%% decomposed büyük olarak görüntüleme
decomposedLogData = residual
decomposedLogData.dropna(inplace=True)
test_stationary(decomposedLogData)
#%%
"""
lag_acf = acf(veriler2logShifting, nlags=20)
lag_pacf = pacf(veriler2logShifting, nlags=20)

#PLOT ACF
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--', color='grey')
plt.axhline(y=-1.96/np.sqrt(len(veriler2logShifting)),linestyle='--',color='grey')
plt.axhline(y=1.96/np.sqrt(len(veriler2logShifting)),linestyle='--',color='grey')
plt.title('autocorelation function')

#PLOT PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--', color='grey')
plt.axhline(y=-1.96/np.sqrt(len(veriler2logShifting)),linestyle='--',color='grey')
plt.axhline(y=1.96/np.sqrt(len(veriler2logShifting)),linestyle='--',color='grey')
plt.title('partial autocorelation function')
plt.tight_layout()
"""
#%%
"""  ARIMA MODEL İLE FORECASTİNG  """

model = ARIMA(veriler2_log, order=(2,1,2)) ##RSS i zaaltmak için order=(2,1,0) vb deneyebilirsin.
results_AR = model.fit(disp=-1)
plt.plot(veriler2logShifting)
plt.plot(results_AR.fittedvalues,color='red')
plt.title('RSS: %4f'%sum((results_AR.fittedvalues-veriler2logShifting['Close'])**2))
print('Plotting AR model')

### RSS ne kadar az çıkarsa o kadar iyi
#%%
predictions_ARIMA_diff = pd.Series(results_AR.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())
#%%
"""  convert to cumulative sum  """
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())
#%%
predictions_ARIMA_log = pd.Series(veriler2_log['Close'].ix[0], index=veriler2_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA_log.head()

#%% 
"""  datayı orjinal haline geri getirmek  """
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(veriler2,color='black')
plt.plot(predictions_ARIMA)

#%% 
""" ********* ARIMA İLE FORECAST OLUŞTURMAK  **********  """

#kaç adet timestamp adımım olduğunu tespit ettim.
kaçsatır=len(veriler2_log.index)


#gelecek 1 yıl için tahminde:
# plot_predict(1,forecast edeceğim step sayısı + kaçsatır)
# forecast step sayım=12 olmalı (ay ay timesstamp verisi var)

results_AR.plot_predict(1,kaçsatır+100)

#results_AR.forecast(steps=12)   #SAYISAL OLARAK GELECEKTE OLACAK STEPLERİN TAHMİN SONUÇLARI
 


