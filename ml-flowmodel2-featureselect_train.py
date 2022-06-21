import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import dask
import xarray as xr
from dask.diagnostics import ProgressBar

import sys
print(sys.executable)

def shift_time(ds, value):
    ds.coords['time'].values = pd.to_datetime(ds.coords['time'].values) + value
    return ds

static = xr.open_dataset('../data/usa/era5_slt_z_slor_lsm_stationary_field.nc')

era5 = xr.open_dataset('../data/usa/era5_lsp_cp_1981-2017_daysum.nc')

glofas = xr.open_dataset('../data/usa/glofas_reanalysis_usa_1981-2002.nc')
glofas = glofas.rename({'lat': 'latitude', 'lon': 'longitude'})  # to have the same name like in era5
glofas = shift_time(glofas, -dt.timedelta(days=1))
# da.transpose(dims='latitude')  # flip?

glofas_rerun = xr.open_dataset('../data/glofas-freruns/2013051800/glofas2.3_era5wb_reforecast_dis_bigchannels_1000km2_20130518_0.nc')
glofas_rerun = glofas_rerun.rename({'lat': 'latitude', 'lon': 'longitude'})
glofas_rerun = shift_time(glofas_rerun, -dt.timedelta(days=1))

era5 = era5.isel(time=slice(0*365,5*365))
glofas = glofas.isel(time=slice(0*365,5*365))

if len(era5.time) < 3000:
    era5 = era5.load()
    glofas = glofas.load()

glofas.coords

glofas = glofas.sel(latitude=slice(40, 28), #28, 40),
                    longitude=slice(-95, -85))

if not 'lsp' in era5:
    lsp = era5['tp']-era5['cp']
    lsp.name = 'lsp'
else:
    lsp = era5['lsp']

reltop = era5['z'].sel(level=500) - era5['z'].sel(level=850)
reltop.name = 'reltop'

q_mean = era5['q'].mean('level')
q_mean.name = 'q_mean'

era5 = xr.merge([era5['cp'], lsp, reltop, q_mean])

dis = glofas['dis']
X_dis = glofas

z_glofas = static['z'].isel(time=0)/9.81  # converting to m approx.
z_glofas = z_glofas.interp(latitude=glofas.latitude,
                           longitude=glofas.longitude)
z_glofas = z_glofas.drop('time')  # time is misleading as the topography does not change

a = dis.max('time')
plt.imshow(a)

a = a[70,:]  # i want a gridpoint at this latitude
np.where(a==a.max()) #.sel(longitude=87).values =.max()

point = dict(latitude=70, longitude=38)
box = dict(latitude=slice(point['latitude']-10,
                          point['latitude']+10),
           longitude=slice(point['longitude']-10,
                           point['longitude']+10),)

X_dis = dis.isel(box)
y_dis = dis.isel(point)

fig, ax = plt.subplots()
dis.max('time').plot(ax=ax)
ax.plot(y_dis.longitude, y_dis.latitude, color='cyan', marker='o', 
        markersize=20, mew=4, markerfacecolor='none')

# rough river map
river = dis.min('time') > 5
river.name = 'river mask [0/1]'
river.plot()

i, j = 70, 38
di = 20
dj = 20
pct = 0.1  # influencing gridpoint must have mean discharge more than this percentage

i0, i1 = i-di, i+di
j0, j1 = j-dj, j+dj

meandis = dis.mean('time')
dis_box_mean = meandis[i0:i1,j0:j1]
z_box = z_glofas[i0:i1,j0:j1]
z_box.name = 'topography at river gridpoints [m]'

def correlate(da_3d, da_timeseries, timelag=False):
    a = da_3d - da_3d.mean('time')
    b = da_timeseries - da_timeseries.mean('time')
    N = len(b.coords['time'])
    if timelag:
        b = b.drop('time')
        a = a.drop('time')
    out = b.dot(a)/a.std('time')/b.std()/N
    out.name = 'correlation coefficient'
    return out

#for ic in range(i0, i1):
#    for jc in range(j0, j1):
cntr = dis[:,i,j]
dis_box = dis[:,i0:i1,j0:j1]
dis_box_relevant = dis_box.where(river==1)
a = correlate(dis_box_relevant, cntr)
a.plot()

lags = [-1, 1]

timelag_corrs = np.full((len(lags), di*2, dj*2), np.nan)
for t, lag in enumerate(lags):
    if lag > 0:  # dis_box with data from previous timesteps
        cntr = dis[lag:,i,j]
        dis_box = dis[:-lag,i0:i1,j0:j1]
    elif lag < 0:  # dis_box with data from future timesteps
        cntr = dis[:lag,i,j]
        dis_box = dis[-lag:,i0:i1,j0:j1]
        
    dis_box_relevant = dis_box.where(river==1)
    timelag_corrs[t,:,:] = correlate(dis_box_relevant, cntr, timelag=True)

lag_influencing = timelag_corrs[1,:,:]>timelag_corrs[0,:,:]
plt.imshow(lag_influencing)

fig, ax = plt.subplots()
dis_box_mean.plot()

fig, ax = plt.subplots()
z_box.where(river==1).plot()

# select feature gridpoints
fig, ax = plt.subplots()
influencer = (dis_box_mean > pct*meandis[i,j])  \
              &(z_box >= z_glofas[i,j])   \
              &(river==1) & lag_influencing
influencer.name = 'gridpoints influencing discharge [0/1]'
influencer.plot()

def add_shifted_predictors(ds, shifts, variables='all'):
    """Adds additional variables to an array which are shifted in time.
    
    Parameters
    ----------
    ds : xr.Dataset
    shifts : list of integers
    variables : str or list
    """
    if variables == 'all': 
        variables = ds.data_vars
        
    for var in variables:
        for i in shifts:
            if i == 0: continue  # makes no sense to shift by zero
            newvar = var+'-'+str(i)
            ds[newvar] = ds[var].shift(time=i)
    return ds

X_dis = glofas.where(influencer)

# add (t-2) and (t-3) as additional features
shifts = range(1,4)
X_dis = add_shifted_predictors(X_dis, shifts, variables='all')
X_dis = X_dis.drop('dis')  # we actually want to predict (t) with (t-1, t-2, t-3)
X_dis

# what currently is time, is in fact the number of samples we have
#X_atm = X_atm.rename(time='samples')  

# the different variables encode time features
X_dis = X_dis.to_array(dim='time_feature')  
X_dis

# our model needs spatial and time information as features, so we need to reshape
X_dis = X_dis.stack(features=['latitude', 'longitude', 'time_feature'])
#X_atm = X_atm.drop(['latitude', 'longitude', 'time_feature'])

X_dis

shifts = range(1,11)
notshift_vars = ['swvl1', 'swvl2']
shift_vars = [v for v in X.data_vars if not v in notshift_vars]

Xs = add_shifted_predictors(X, shifts, variables=shift_vars)

Xar = X_dis
yar = dis[:,i,j]

yar = yar.drop(['latitude', 'longitude'])
#yar = yar.rename({'time': 'samples'})  # rename only to allow for concat to work to synchronize arrays
yar.coords['features'] = 'dis' #yar.expand_dims('features').T

# remove features that are nan
Xar = Xar.dropna('features', how='all')

Xar.shape

yar.shape

Xy = xr.concat([Xar, yar], dim='features')
Xy

Xyt = Xy.dropna('time', how='any')  # drop them as we cannot train on nan values
time = Xyt.time

assert len(Xyt.time) > 1

Xyt

Xda = Xyt[:,:-1]
yda = Xyt[:,-1]

Xda = Xda.chunk(dict(time=-1, features=-1)).data
yda = dask.array.from_array(yda.data.squeeze())

print('Xda.shape:', Xda.shape, 'yda.shape:',  yda.shape)

print(Xda, yda)

import joblib
from sklearn.pipeline import Pipeline
from dask_ml.preprocessing import StandardScaler
from dask_ml.decomposition import PCA

#from dask_ml.xgboost import XGBRegressor
#from dask_ml.linear_model import LogisticRegression
from dask_ml.linear_model import LinearRegression

import keras
from keras.layers.core import Dropout

class KerasDenseNN(object):
    def __init__(self, **kwargs):
        model = keras.models.Sequential()
        self.cfg = kwargs
        ha = self.cfg.get('hidden_activation')

        for N_nodes in self.cfg.get('N_hidden_nodes'):
            # activation -> solve the dying ReLU problem
            # https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks
            hidden = keras.layers.Dense(N_nodes)
            #            bias_initializer=self.cfg.get('bias_initializer'))

            model.add(hidden)
            model.add(keras.layers.Activation('tanh')) #keras.layers.ReLU())
            
            if self.cfg.get('dropout'):
                model.add(Dropout(self.cfg.get('dropout')))

        outputlayer = keras.layers.Dense(1, activation='linear')
        model.add(outputlayer)

        optimizer_name, options_dict = self.cfg.get('optimizer')
        optimizer = getattr(keras.optimizers, optimizer_name)(**options_dict)

        model.compile(loss=self.cfg.get('loss'), optimizer=optimizer)
        self.model = model

        self.callbacks = [keras.callbacks.EarlyStopping(monitor='loss',
                    min_delta=0.001, patience=100, verbose=0, mode='auto',
                    baseline=None, restore_best_weights=True),]

    def predict(self, X):
        return self.model.predict(X).squeeze()

    def fit(self, X, y, **kwargs):
        return self.model.fit(X, y,
                              epochs=self.cfg.get('epochs', None),
                              batch_size=self.cfg.get('batch_size', None),
                              callbacks=self.callbacks,
                              verbose=1,
                              **kwargs)

mlp_kws = dict(optimizer=('RMSprop', dict(lr=0.01)),
               loss='mean_squared_error',
               N_hidden_nodes=(4,4,),
               hidden_activation=keras.layers.Activation('tanh'), #keras.layers.ReLU(), #-LeakyReLU(alpha=0.3), #'relu',
               output_activation='linear',
               bias_initializer='zeros',
               batch_size=512,
               dropout=.0, #25,
               epochs=1000,
              )


linear_kws = dict(C=.1, n_jobs=-1, max_iter=10000, verbose=True)

if False:
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('pca', PCA(n_components=4)),
                     ('model', LinearRegression(**linear_kws)),],
                    verbose=True)
if True:
    pipe = Pipeline([('scaler', StandardScaler()),
                     #('pca', PCA(n_components=4)),
                     ('model', KerasDenseNN(**mlp_kws)),],
                    verbose=True)

pipe

#use_keras = 'Keras' in str(pipe.named_steps['model'].__class__)
#Xda_is_dask = isinstance(Xda, dask.array.core.Array)
#if use_keras and Xda_is_dask:
try:
    
    print('loading data...')
    Xda = Xda.persist()  # if loaded, stay loaded, dont clean
    Xda = Xda.compute()  # load data into RAM (convert to numpy array)
    yda = yda.compute()
except:
    pass

print(Xda.shape, yda.shape)

history = pipe.fit(Xda, yda)

keras.utils.print_summary(pipe.named_steps['model'].model)

h = history.named_steps['model'].model.history

# Plot training & validation loss values
plt.plot(h.history['loss'])
#plt.plot(h.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.gca().set_yscale('log')
plt.show()

def add_time(vector, time, name=None):
    """Converts arrays to xarrays with a time coordinate."""
    return xr.DataArray(vector, dims=('time'), coords={'time': time}, name=name)

with ProgressBar():
    ytest = pipe.predict(Xda)

ytest.shape

ytest = add_time(ytest, time, name='dis-forecast')
ytest_dis = ytest #.cumsum('time')
ytest_dis.values
#ytest_dis += y[0]  # initial state + changes = timeseries of forecasted discharge

fig, ax = plt.subplots(figsize=(24,5))
obs = dis[:,i,j].to_pandas()
fcst = ytest_dis.to_pandas()
obs.plot(ax=ax, label='dis-reanalysis')
fcst.plot(ax=ax, label='dis-forecast')
plt.legend()

(fcst-obs).plot()

