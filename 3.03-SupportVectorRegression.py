import sys
sys.path.append("../../")
print(sys.executable)
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import dask
dask.config.set(scheduler='threads')
import xarray as xr

from mlcombimodel1loop import add_time, generate_prediction_array, remove_outlier, multi_forecast_case_study
#from python.aux.utils_floodmodel import add_time, generate_prediction_array, remove_outlier, multi_forecast_case_study
from python.aux.plot import plot_multif_prediction

import joblib
from sklearn.pipeline import Pipeline
from dask_ml.preprocessing import StandardScaler
from dask_ml.decomposition import PCA

from sklearn.svm import SVR

import matplotlib
matplotlib.rcParams.update({'font.size': 14})

# load data
features = xr.open_dataset('../../data/features_xy.nc')
y = features['dis']
X = features.drop(['dis', 'dis_diff'])

features

dis_shift_switch = False

abs_vals_switch = False

if dis_shift_switch:
    dis_shift_1 = y.shift(time=1)
    X = X.assign({'dis-1': dis_shift_1})

X_base = X.to_array(dim='features').T.copy()
y_base = y.copy()

period_train = dict(time=slice(None, '2005'))
period_valid = dict(time=slice('2006', '2011'))
period_test = dict(time=slice('2012', '2016'))

X_train, y_train = X_base.loc[period_train], y_base.loc[period_train]
X_valid, y_valid = X_base.loc[period_valid], y_base.loc[period_valid]
X_test, y_test = X_base.loc[period_test], y_base.loc[period_test]

X_train.shape, y_train.shape

time = y_train.time
Xda = X_train.chunk(dict(time=-1, features=-1)).dropna(dim='time').to_pandas()

if abs_vals_switch:
    # train on absolute values
    yda = y_train.to_pandas().loc[Xda.index]
    # removing outlier and removing same parts from X
    yda = remove_outlier(yda)
    Xda = Xda.loc[yda.index]
else:
    # train on change in discharge values
    yda = y_train.diff(dim='time').to_pandas().loc[Xda.index]
    # removing outlier and removing same parts from X
    yda = remove_outlier(yda)
    Xda = Xda.loc[yda.index]

model = SVR(kernel='poly',
            C=100,
            epsilon=0.01,
            degree=3,
           )

pipe = Pipeline([('scaler', StandardScaler()),
                 #('pca', PCA(n_components=6)),
                 ('model', model),], verbose=True)

x_fit = Xda.copy()
y_fit = yda.copy()
pipe.fit(x_fit, y_fit)

Xda_insample = Xda.copy()
insample_check = pipe.predict(Xda_insample)
insample_check = add_time(insample_check, Xda_insample.index, name='forecast')
insample_check.where(insample_check > insample_check.quantile(0.01)).to_pandas().plot(linewidth=0.5)
yda.plot(linestyle='--', linewidth=0.5)

# prediction start from every nth day
# if in doubt, leave n = 1 !!!
n = 1
X_pred = X_valid[::n].copy()
y_pred = pipe.predict(X_pred.values)
y_pred = add_time(y_pred, X_pred.time, name='forecast')
print(y_pred)

import matplotlib
matplotlib.rcParams.update({'font.size': 14})
multif = generate_prediction_array(y_pred, y, forecast_range=14)
plot_multif_prediction(multif, y, forecast_range=14, title='14-day forecast - Validation period - SupportVectorRegression');
plt.savefig('validation_period_svr.png', dpi=600, bbox_inches='tight')

forecast_range = 14
y_o_pers = y_valid
# persistence
y_m_pers = y_valid.copy()
for i in range(1, forecast_range):
    y_m_pers.loc[y_valid.time[i::forecast_range]] = y_valid.shift(time=i)[i::forecast_range].values

rmse = np.sqrt(np.nanmean((y_m_pers - y_o_pers)**2))
nse = 1 - np.sum((y_m_pers - y_o_pers)**2)/(np.sum((y_o_pers - np.nanmean(y_o_pers))**2))
print(f"Persistence {forecast_range}-day forecast: RMSE={round(float(rmse), 2)}; NSE={round(float(nse.values), 2)}")

# prediction start from every nth day
# if in doubt, leave n = 1 !!!
n = 1
X_pred = X_test[::n].copy()
y_pred = pipe.predict(X_pred)
y_pred = add_time(y_pred, X_pred.time, name='forecast')

multif_test = generate_prediction_array(y_pred, y, forecast_range=14)
plot_multif_prediction(multif_test, y, forecast_range=14, title='SupportVectorRegression: kernel=poly C=100; epsilon=0.01, degree=3');

X_multif_fin, X_multifr_fin, y_case_fin = multi_forecast_case_study(pipe, X_test, y)

fig, ax = plt.subplots(figsize=(15, 5))
frerun_c = ['silver', 'darkgray', 'gray', 'dimgray']

y_case_fin.to_pandas().plot(ax=ax, label='reanalysis', linewidth=4)
run = 0
for i in X_multifr_fin.num_of_forecast:
    X_multif_fin.sel(num_of_forecast=i).to_pandas().T.plot(ax=ax, label='forecast',
                                                           linewidth=2, color='tab:purple')
    X_multifr_fin.sel(num_of_forecast=i).to_pandas().T.plot(ax=ax, label='frerun', linewidth=1,
                                                            linestyle='--', color=frerun_c[run])
    run += 1
ax.set_ylabel('river discharge [m$^3$/s]')
plt.legend(['reanalysis', 'support vector regression', 'GloFAS 05-18', 'GloFAS 05-22', 'GloFAS 05-25', 'GloFAS 05-29'])
plt.title('SupportVectorRegression: kernel=poly C=100; epsilon=0.01; degree=3; case study May/June 2013');
plt.savefig('svr_case.png', dpi=600, bbox_inches='tight')

X_multif_fin.to_netcdf('../../data/models/SVR/support_vector_regression_result_case_study.nc', mode='w')

multif_test.to_netcdf('../../data/models/SVR/support_vector_regression_result_test_period.nc', mode='w')


