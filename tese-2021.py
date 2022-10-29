# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 14:05:13 2022

@author: Tomas
"""


import pandas as pd
import numpy as np
from matplotlib import pyplot
import pmdarima as pmd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import arch

data = pd.read_excel('data-inf2.xlsx',index_col=0,parse_dates = True)
data_diff=data.diff().dropna()
for i in range(len(data_diff.columns)-1):
    max_lags=4
    y='GP'
    results = grangercausalitytests(data_diff[[y,data_diff.columns[i+1]]],max_lags, verbose=False)
    p_values = [round(results[i+1][0]['ssr_ftest'][1],4) for i in range(max_lags)]
    print('Column - {} : P-Values - {}'.format(data_diff.columns[i+1],p_values))
    
data_train = data[:int(0.995*len(data))]

data_test = data[int(0.995*len(data)):]

stepwise_fit_10y = pm.auto_arima(data_train.GP,exogenous=data_train[['10-YY']], max_d=1, trace=True, error_action='ignore', information_criterion='aic', # don't want to know if an order does not work
suppress_warnings=True, # don't want convergence warnings
stepwise=True)
print(stepwise_fit_10y.summary())
future_forecast_10y=stepwise_fit_10y.predict(X=data_test[['10-YY']],n_periods=len(data_test))



stepwise_fit_dol = pm.auto_arima(data_train.GP,exogenous=data_train[['DOL']], max_d=1,trace=True, error_action='ignore', information_criterion='aic', # don't want to know if an order does not work
suppress_warnings=True, # don't want convergence warnings
stepwise=True)
print(stepwise_fit_dol.summary())
future_forecast_dol=stepwise_fit_dol.predict(X=data_test[['DOL']],n_periods=len(data_test),alpha=0.01)

stepwise_fit_4var = pm.auto_arima(data_train.GP,exogenous=data_train[['10-YY','DOL','RIR','INF']], max_d=1,trace=True, error_action='ignore', information_criterion='aic', # don't want to know if an order does not work
suppress_warnings=True, # don't want convergence warnings
stepwise=True)
print(stepwise_fit_4var.summary())
future_forecast_4var=stepwise_fit_4var.predict(X=data_test[['10-YY','DOL','RIR','INF']],n_periods=len(data_test))

stepwise_fit_inf = pm.auto_arima(data_train.GP,exogenous=data_train[['INF']], trace=True, error_action='ignore', information_criterion='aic', # don't want to know if an order does not work
suppress_warnings=True, # don't want convergence warnings
stepwise=True)
print(stepwise_fit_inf.summary())
future_forecast_inf=stepwise_fit_inf.predict(X=data_test[['INF']],n_periods=len(data_test))
    
    
stepwise_fit_rir = pm.auto_arima(data_train.GP,exogenous=data_train[['RIR']], max_d=1,trace=True, error_action='ignore', information_criterion='aic', # don't want to know if an order does not work
suppress_warnings=True, # don't want convergence warnings
stepwise=True)
print(stepwise_fit_rir.summary())
future_forecast_rir=stepwise_fit_rir.predict(X=data_test[['RIR']],n_periods=len(data_test))

from sklearn.metrics import mean_squared_error
from pmdarima.metrics import smape

print(f"Mean squared error 10y: {mean_squared_error(data_test.GP, future_forecast_10y)}")
print(f"SMAPE: {smape(data_test.GP, future_forecast_10y)}")

    
print(f"Mean squared error dol: {mean_squared_error(data_test.GP, future_forecast_dol)}")
print(f"SMAPE: {smape(data_test.GP, future_forecast_dol)}")

    
print(f"Mean squared error 4var: {mean_squared_error(data_test.GP, future_forecast_4var)}")
print(f"SMAPE: {smape(data_test.GP, future_forecast_4var)}")

print(f"Mean squared error inf: {mean_squared_error(data_test.GP, future_forecast_inf)}")
print(f"SMAPE: {smape(data_test.GP, future_forecast_inf)}")

print(f"Mean squared error rir: {mean_squared_error(data_test.GP, future_forecast_rir)}")
print(f"SMAPE: {smape(data_test.GP, future_forecast_rir)}")


def rsquared(x,y):
    SSres=0
    SStot=0
    media= x.mean()
    for i in range(len(x)):
        SSres+=(x[i]-y[i])**2
        SStot+=(x[i]-media)**2
    return 1-SSres/SStot



"""from statsmodels.tsa.stattools import adfuller
test1 = adfuller(gdp_train)
print("p-value =", test1[1])

if test1[1]> 0.05:
    print("We do not reject the H0: non-stationary time series <-> we have a unit root")"""