import sys
sys.path.append("../../")
import numpy as np
import datetime as dt

import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import pandas as pd
import seaborn as sns
import xarray as xr
import keras

#from python.aux.utils_floodmodel import add_time, generate_prediction_array, remove_outlier, multi_forecast_case_study
from python.aux.plot import plot_multif_prediction

# load data
ds = xr.open_dataset('../../data/features_xy.nc')

y_orig = ds['dis']
y = y_orig.copy()
X = ds.drop(['dis', 'dis_diff'])

X

from scipy.stats import moment
dist = y.values.ravel() - y.mean().values.ravel()
dist = dist[~np.isnan(dist)]
sns.distplot(dist)
print('mean:', moment(dist, 1), ', std:', moment(dist, 2), ', skew:', moment(dist, 3))

from scipy.stats import moment
dist = y.diff('time', n=1).values.ravel()
dist = dist[~np.isnan(dist)]
sns.distplot(dist)
print('mean:', moment(dist, 1), ', std:', moment(dist, 2), ', skew:', moment(dist, 3))

y = y.diff('time', 1)
y

from python.aux.utils_floodmodel import reshape_scalar_predictand

Xda, yda = reshape_scalar_predictand(X, y)
Xda.features

period_train = dict(time=slice(None, '2005'))
period_valid = dict(time=slice('2006', '2011'))
period_test = dict(time=slice('2012', '2016'))

X_train, y_train = Xda.loc[period_train], yda.loc[period_train]
X_valid, y_valid = Xda.loc[period_valid], yda.loc[period_valid]
X_test, y_test = Xda.loc[period_test], yda.loc[period_test]

X_train.shape, y_train.shape

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import keras
from keras.layers.core import Dropout
from keras.constraints import MinMaxNorm, nonneg


def add_time(vector, time, name=None):
    """Converts numpy arrays to xarrays with a time coordinate.

    Parameters
    ----------
    vector : np.array
        1-dimensional array of predictions
    time : xr.DataArray
        the return value of `Xda.time`

    Returns
    -------
    xr.DataArray
    """
    return xr.DataArray(vector, dims=('time'), coords={'time': time}, name=name)


class DenseNN(object):
    def __init__(self, **kwargs):
        self.xscaler = StandardScaler()
        self.yscaler = StandardScaler()

        model = keras.models.Sequential()
        self.cfg = kwargs
        hidden_nodes = self.cfg.get('hidden_nodes')

        model.add(keras.layers.Dense(hidden_nodes[0],
                                     activation='tanh'))
        model.add(keras.layers.BatchNormalization())
        model.add(Dropout(self.cfg.get('dropout', None)))

        for n in hidden_nodes[1:]:
            model.add(keras.layers.Dense(n, activation='tanh'))
            model.add(keras.layers.BatchNormalization())
            model.add(Dropout(self.cfg.get('dropout', None)))
        model.add(keras.layers.Dense(self.output_dim,
                                     activation='linear'))
        opt = keras.optimizers.Adam()

        model.compile(loss=self.cfg.get('loss'), optimizer=opt)
        self.model = model

        self.callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                            min_delta=1e-2, patience=100, verbose=0, mode='auto',
                            baseline=None, restore_best_weights=True),]

    def score_func(self, X, y):
        """Calculate the RMS error

        Parameters
        ----------
        xr.DataArrays
        """
        ypred = self.predict(X)
        err_pred = ypred - y

        # NaNs do not contribute to error
        err_pred = err_pred.where(~np.isnan(err_pred), 0.)
        return float(np.sqrt(xr.dot(err_pred, err_pred)))

    def predict(self, Xda, name=None):
        """Input and Output: xr.DataArray

        Parameters
        ----------
        Xda : xr.DataArray
            with coordinates (time,)
        """
        X = self.xscaler.transform(Xda.values)
        y = self.model.predict(X).squeeze()
        y = self.yscaler.inverse_transform(y)

        y = add_time(y, Xda.time, name=name)
        return y

    def fit(self, X_train, y_train, X_valid, y_valid, **kwargs):
        """
        Input: xr.DataArray
        Output: None
        """

        print(X_train.shape)
        X_train = self.xscaler.fit_transform(X_train.values)
        y_train = self.yscaler.fit_transform(
                        y_train.values.reshape(-1, self.output_dim))

        X_valid = self.xscaler.transform(X_valid.values)
        y_valid = self.yscaler.transform(
                        y_valid.values.reshape(-1, self.output_dim))

        return self.model.fit(X_train, y_train,
                              validation_data=(X_valid, y_valid),
                              epochs=self.cfg.get('epochs', 1000),
                              batch_size=self.cfg.get('batch_size'),
                              callbacks=self.callbacks,
                              verbose=0, **kwargs)

config = dict(hidden_nodes=(64,),  
                dropout=0.25,
                epochs=300,
                batch_size=90,
                loss='mse')

m = DenseNN(**config)

hist = m.fit(X_train, y_train, X_valid, y_valid)

m.model.summary()

from keras.utils import plot_model
plot_model(m.model, to_file='model.png', show_shapes=True)

h = hist.model.history

# Plot training & validation loss value
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(h.history['loss'], label='loss')
ax.plot(h.history['val_loss'], label='val_loss')
plt.title('Learning curve')
ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')
plt.legend(['Training', 'Validation'])
ax.set_yscale('log')

import os, yaml
dir_model = '../../models/tdnn-diff-batch90/'
os.makedirs(dir_model, exist_ok=True)

yaml_string = m.model.to_yaml()
with open(dir_model+'keras-config.yml', 'w') as f:
    yaml.dump(yaml_string, f)

with open(dir_model+'model-config.yml', 'w') as f:
    yaml.dump(config, f, indent=4)

from contextlib import redirect_stdout
with open(dir_model+'summary.txt', "w") as f:
    with redirect_stdout(f):
        m.model.summary()

y_pred_train = m.predict(X_train)
y_pred_train = generate_prediction_array(y_pred_train, y_orig, forecast_range=14)

y_pred_valid = m.predict(X_valid)
y_pred_valid = generate_prediction_array(y_pred_valid, y_orig, forecast_range=14)

y_pred_test = m.predict(X_test)
y_pred_test = generate_prediction_array(y_pred_test, y_orig, forecast_range=14)

title='Setting: Time-Delay Neural Net: 64 hidden nodes, dropout 0.25'
plot_multif_prediction(y_pred_test, y_orig, forecast_range=14, title=title);

from python.aux.utils_floodmodel import multi_forecast_case_study_tdnn

X_multif_fin, X_multifr_fin, y_case_fin = multi_forecast_case_study_tdnn(m)

X_multif_fin.plot()

fig, ax = plt.subplots(figsize=(15, 5))
color_scheme = ['g', 'cyan', 'magenta', 'k']

y_case_fin.to_pandas().plot(ax=ax, label='reanalysis', lw=4)
run = 0
for i in X_multifr_fin.num_of_forecast:
    X_multif_fin.sel(num_of_forecast=i).to_pandas().T.plot(ax=ax,
                                                           label='forecast',
                                                           linewidth=2,
                                                           color='firebrick')
    X_multifr_fin.sel(num_of_forecast=i).to_pandas().T.plot(ax=ax,
                                                            label='frerun',
                                                            linewidth=0.9,
                                                            linestyle='--',
                                                            color=color_scheme[run])
    run += 1
ax.set_ylabel('river discharge [m$^3$/s]')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='b', lw=4),
                Line2D([0], [0], color='firebrick', lw=2),
                Line2D([0], [0], color='g', linestyle='--'),
                Line2D([0], [0], color='cyan', linestyle='--'),
                Line2D([0], [0], color='magenta', linestyle='--'),
                Line2D([0], [0], color='k', linestyle='--')]

legendlabels = ['reanalysis', 'neural net', 'EFAS 05-18', 'EFAS 05-22', 'EFAS 05-25', 'EFAS 05-29']
ax.legend(custom_lines, legendlabels, fontsize=11)

plt.title('Setting: Time-Delay Neural Net: 64 hidden nodes, dropout 0.25');

y_pred_valid.to_netcdf(dir_model+'tdnn_result_validation_period.nc')
y_pred_test.to_netcdf(dir_model+'tdnn_result_test_period.nc')
X_multif_fin.to_netcdf(dir_model+'tdnn_result_case_study.nc')

import eli5
from eli5.permutation_importance import get_score_importances

base_score, score_decreases = get_score_importances(m.score_func, X_test, y_test, n_iter=5)

importances = score_decreases/np.max(score_decreases)  # normalize

def feature_importance_plot(xda_features, importances):
    # xda_features : xr.DataArray.features
    # score_decreases : list of arrays
    mmin, mmed, mmax = np.min(importances, axis=0), np.median(importances, axis=0), np.max(importances, axis=0)

    labels = [e[0] for e in xda_features.values]
    assert len(labels) == len(importances[0])  # one label per entry
    plt.subplots(figsize=(15,5))

    x = np.arange(len(labels))
    plt.bar(x, mmed, yerr=(mmed-mmin, mmax-mmed), width = .8, ecolor='black', capsize=8)

    plt.xticks(ticks=x, labels=labels, rotation=20)
    plt.ylabel('normalized importance')
    plt.title('Estimate of Feature Importance by 5 row-wise Permutations')

feature_importance_plot(Xda.features, importances)

df = pd.read_csv('tdnn-hyperparams4.txt', sep=';', header=0)
df.sort_values('rmse')

