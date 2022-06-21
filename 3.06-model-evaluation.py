import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import seaborn as sns
import xarray as xr

import sys
sys.path.append('../../')
from python.aux.plot import plot_recurrent
from python.aux.utils import xr_to_datetime
from python.aux.utils_floodmodel import add_valid_time
from python.aux.verification import verify, ME, RMSE, RMSE_persistence, NSE, NSE_diff

def multif_metrics(pred, obs, method='total', case=True, name=None):
    """Calculates RMSE and NSE metrics for xr.DataArray in the multiforecast
    shape as defined in the previous notebooks.
    Use case=True if the input data is in the form of a case study: multiple forecasts vor one set period.

    Parameters
    ----------
        pred   : xr.DataArray
        obs    : xr.DataArray
        case   : Boolean
        method : str
    """
    if method == 'total':
        y_o = obs.loc[{'time': pred.time.values.ravel()}].values
        y_m = pred.values.ravel()
        rmse = np.sqrt(np.nanmean((y_m - y_o)**2))
        nse = 1 - np.sum((y_m - y_o)**2)/np.sum(((y_o - np.nanmean(y_o))**2))
        metr = xr.DataArray(data=np.array([rmse, nse]),
                            dims=['metric'],
                            coords=[['rmse', 'nse']],
                            name='total_metrics')

    elif method == 'per_day':
        forecast_range = pred.forecast_day
        fd_list = []
        for fd in forecast_range:
            pred_day = pred.sel(forecast_day=fd)
            y_o = obs.loc[{'time': pred_day.time.values.ravel()}].values
            y_m = pred_day.values.ravel()
            rmse = np.sqrt(np.nanmean((y_m - y_o)**2))
            nse = 1 - np.sum((y_m - y_o)**2)/np.sum(((y_o - np.nanmean(y_o))**2))
            fd_list.append([rmse, nse])
        metr = xr.DataArray(data=np.array(fd_list),
                            dims=['forecast_day', 'metric'],
                            coords=[range(np.array(fd_list).shape[0]), ['rmse', 'nse']],
                            name='per_day_metrics')
    if name:
        metr.name = f"{name}-{metr.name}"
    return metr

truth_test = xr.open_dataset('../../data/glofas_reanalysis_test_period.nc')['dis']
fc_lr_test = xr.open_dataset('../../data/models/LinearRegression/linear_regression_result_test_period.nc')['prediction']
fc_svr_test = xr.open_dataset('../../data/models/SVR/support_vector_regression_result_test_period.nc')['prediction']
fc_gb_test = xr.open_dataset('../../data/models/GradientBoost/gradient_boost_result_test_period.nc')['prediction']
fc_tdnn_test = xr.open_dataset('../../data/models/TimeDelayNeuralNet/tdnn_result_test_period.nc')['prediction']

# initial state used for persistence
truth_init = truth_test.values.reshape(fc_lr_test.shape)[:,0]
# repeat the initial state for the whole forecast period i.e. persistence forecast
persistence_data = np.tile(truth_init, (15, 1)).transpose()
# easy way to get the same shape, dims, coords etc
persistence_test = fc_lr_test.copy()
persistence_test.values = persistence_data

lr_test = multif_metrics(fc_lr_test, truth_test, method='per_day', name='LinReg')
svr_test = multif_metrics(fc_svr_test, truth_test, method='per_day', name='SVR')
gb_test = multif_metrics(fc_gb_test, truth_test, method='per_day', name='GradBoost')
tdnn_test = multif_metrics(fc_tdnn_test, truth_test, method='per_day', name='TimeDelay_neural_net')
per_test = multif_metrics(persistence_test, truth_test, method='per_day', name='Persistence')

list_of_models = [lr_test, svr_test, gb_test, tdnn_test, per_test]

