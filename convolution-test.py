import numpy as np
import datetime as dt
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import keras

n = 1000
t = np.linspace(0,9*np.pi,n)
x = np.zeros(n)

t.shape, x.shape

mask = (t>np.pi)&(t<2*np.pi)
x[mask] += np.sin(t[mask])**2

mask = (t>3*np.pi)&(t<4*np.pi)
x[mask] += np.sin(t[mask])**2

plt.plot(x)

#mask = (t>3*np.pi/2)&(t<2*np.pi)
#x[mask] += np.sin(2*t[mask])**2/2


mask = (t>2*np.pi)&(t<3*np.pi)
x[mask] += np.sin(t[mask])**2/2

mask = (t>4*np.pi)&(t<5*np.pi)
x[mask] += np.sin(t[mask])**2/2

plt.plot(x)

y = x.copy()
for i in range(1,len(y)):
    y[i] = y[i-1] - y[i-1]/100 +  x[i]/2

plt.plot(y)

from keras.layers import Dense
from keras.models import Input, Model
from keras.preprocessing.sequence import TimeseriesGenerator
from tcn import TCN

batch_size = None
timesteps = 30
input_dim = 1

i = Input(batch_shape=(batch_size, timesteps, input_dim))

o = TCN(return_sequences=False)(i)  # The TCN layers are here.
o = Dense(1)(o)

m = Model(inputs=[i], outputs=[o])
m.compile(optimizer='adam', loss='mse')

x = x[:,np.newaxis]  # only one predictand so far
x.shape

length = timesteps
data_gen = TimeseriesGenerator(x, y, length, batch_size=x.shape[0])
Xt, yt = data_gen[0]

if len(y.shape)==1:
        yt = yt.reshape(-1, 1)

        Xt.shape, yt.shape

plt.plot(Xt[:,:,0])

m.fit(Xt, yt, epochs=30, validation_split=0.2)

model = keras.models.Sequential()


kernel_size=10
filters=10  # -> full input = 100 length

model.add(keras.layers.Conv1D(filters, kernel_size, input_shape=(100, 1)))


# opt = keras.optimizers.Adadelta(lr=1, rho=0.95, epsilon=0.5, decay=0.0)
# opt = keras.optimizers.RMSprop()
# opt = keras.optimizers.SGD()
opt = keras.optimizers.Adam()

model.compile(loss='mse', optimizer=opt)

yp = m.predict(Xt)
yp.shape

plt.plot(t, y)
plt.plot(t[timesteps:], yp)


