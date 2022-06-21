import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import xarray as xr
import sys
sys.path.append('../')
import dask
dask.config.set(scheduler='processes')

from dask.diagnostics import ProgressBar
#import link_src
from ml_flood.python.misc.plot import Map

xar = xr.open_mfdataset('../data/usa/monthly_files/*precip*.nc')

xar.data_vars

m = Map(figure_kws=dict(figsize=(10,8)))

with ProgressBar():
    fig, ax = m.plot(xar['lsp'].mean('time'))
    ax.set_title('Large-scale precipitation - Mean over time')

with ProgressBar():
    fig, ax = m.plot(xar['lsp'].std('time'))
    ax.set_title('Large-scale precipitation - Standard Deviation over time')

with ProgressBar():
    fig, ax = m.plot(xar['cp'].mean('time'))
    ax.set_title('Convective precipitation - Mean over time')

with ProgressBar():
    fig, ax = m.plot(xar['cp'].std('time'))
    ax.set_title('Convective precipitation - Standard Deviation over time')

anom = xar - xar.mean('time')
da = anom['lsp']

point = dict(latitude=48, longitude=-100)
cov = da.loc[point].dot(da)
cov.plot.pcolormesh()

points = [dict(latitude=48, longitude=-100),
          dict(latitude=47, longitude=-80),
          dict(latitude=38, longitude=-115),
         ]

for point in points:
    cov = da.loc[point].dot(da)/da.var('time')
    with ProgressBar():
        fig, ax = m.plot(cov)
        m.plot_point(ax, lat=point['latitude'], lon=point['longitude'])
        ax.set_title(f"Spatial correlation with N {point['latitude']} E {point['longitude']}")
        plt.show()

da = (xar['lsp'] + xar['cp']).sel(time=slice(dt.datetime(2005,8,1),
                                             dt.datetime(2005,10,1)))
da['long_name'] = 'total precipitation'
da['units'] = 'm'

dask.config.set(scheduler='synchronous')
da = da.load()

da = da.where(da>0.001, 0.)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import xarray as xr

from matplotlib import animation, rc
from IPython.display import HTML

from python.aux.plot import choose_proj_from_xar
from python.aux.ml_flood_config import path_to_data

major_basins_shapefile = path_to_data+'/drainage_basins/Major_Basins_of_the_World.shp'


def create_animation(dataarray):
    da = dataarray
    fig, ax = plt.subplots(figsize=(12,8))
    proj = choose_proj_from_xar(da)
    ax = plt.axes(projection=proj)
    transform = ccrs.PlateCarree()

    countries = cfeature.NaturalEarthFeature(
                            category='cultural',
                            name='admin_0_boundary_lines_land',
                            scale='50m',
                            facecolor='none')
    rivers = cfeature.NaturalEarthFeature(scale='50m', category='physical',
                                          name='rivers_lake_centerlines',
                                          edgecolor='blue', facecolor='none')

    ax.add_feature(countries, edgecolor='grey')
    ax.coastlines('50m')
    ax.add_feature(rivers, edgecolor='blue')

    sf = Reader(major_basins_shapefile)
    shape_feature = ShapelyFeature(sf.geometries(), transform, edgecolor='black')


ax.add_feature(shape_feature, facecolor='none', edgecolor='green')

im = (da.isel(time=0)*np.nan).plot.pcolormesh(ax=ax, transform=transform,
                                                subplot_kws=dict(projection=proj),
                                                cbar_kwargs=dict(fraction=0.025),
                                                vmin=0, vmax=.1,
                                                cmap='ocean_r')
def init():
    im.set_array(np.array([]))
    return im,

def animate(i):
    data = da.isel(time=i)
    time = pd.to_datetime(data.time.values)
    title = str(data.long_name.values)+' - '+time.strftime('%Y-%m-%d')
    im.set_array(data.values.ravel())
    ax.set_title(title)
    return (im,)

    return animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=len(da.coords['time']), interval=100,
                                    blit=True)

anim = create_animation(da)
    
HTML(anim.to_html5_video())

anim.save('tp_2005_August+September.mp4')

HTML(anim.to_jshtml())

