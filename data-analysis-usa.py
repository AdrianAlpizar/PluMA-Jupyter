import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import xarray as xr

import dask
dask.config.set(scheduler='processes')

from dask.diagnostics import ProgressBar
import link_src
from aux.plot import Map

xar = xr.open_mfdataset('../data/usa/*precip*.nc')

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

with ProgressBar():
    fig, ax = m.plot(xar['cp'].std('time')/xar['cp'].mean('time'))
    ax.set_title('Convective precipitation - Relative Standard Deviation over time')

with ProgressBar():
    fig, ax = m.plot(xar['lsp'].std('time')/xar['lsp'].mean('time'))
    ax.set_title('Large-scale precipitation - Relative Standard Deviation over time')

anom = xar - xar.mean('time')
da = anom['lsp']

point = dict(latitude=48, longitude=-100)
cov = da.loc[point].dot(da)
cov.plot.pcolormesh()

points = [dict(latitude=48, longitude=-100),
          dict(latitude=47, longitude=-80),
          dict(latitude=40, longitude=-115),
          dict(latitude=38, longitude=-115),
          dict(latitude=38, longitude=-115)
         ]

for point in points:
    cov = da.loc[point].dot(da)/(da.std('time')*da.loc[point].std())/len(da.coords['time'])
    with ProgressBar():
        fig, ax = m.plot(cov, vmin=-1, vmax=1, cmap='coolwarm_r')
        m.plot_point(ax, lat=point['latitude'], lon=point['longitude'])
        ax.set_title(f"Spatial correlation with N {point['latitude']} E {point['longitude']}")
        plt.show()

data = xar['lsp'].sum('time').values.ravel()
sb.distplot(data)

data = xar['cp'].sum('time').values.ravel()
sb.distplot(data)

x, y = xar['cp'][0,:,:], xar['lsp'][0,:,:]

f, ax = plt.subplots()
ax.set(xscale="log", yscale="log")
mask = (x>0.001)*(y>0.001)
x, y = x.where(mask), y.where(mask)
sb.jointplot(x, y, ax=ax)

xar = xr.open_mfdataset('../data/hourly_for_animations/*precip*.nc')
da = (xar['lsp'] + xar['cp'])*1000
da.name = 'total_precipitation'
da = da.sel(time=slice(dt.datetime(2005,8,25), 
                       dt.datetime(2005,10,1)))

dask.config.set(scheduler='synchronous')
da = da.load()

da = da.where(da>0.001, np.nan)

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import xarray as xr


from matplotlib import animation, rc
from IPython.display import HTML

from aux.plot import choose_proj_from_xar
from aux.ml_flood_config import path_to_data

major_basins_shapefile = path_to_data+'/drainage_basins/Major_Basins_of_the_World.shp'

nws_precip_colors = [
    "#04e9e7",  # 0.01 - 0.10 inches
    "#019ff4",  # 0.10 - 0.25 inches
    "#0300f4",  # 0.25 - 0.50 inches
    "#02fd02",  # 0.50 - 0.75 inches
    "#01c501",  # 0.75 - 1.00 inches
    "#008e00",  # 1.00 - 1.50 inches
    "#fdf802",  # 1.50 - 2.00 inches
    "#e5bc00",  # 2.00 - 2.50 inches
    "#fd9500",  # 2.50 - 3.00 inches
    "#fd0000",  # 3.00 - 4.00 inches
    "#d40000",  # 4.00 - 5.00 inches
    "#bc0000",  # 5.00 - 6.00 inches
    "#f800fd",  # 6.00 - 8.00 inches
    "#9854c6",  # 8.00 - 10.00 inches
    "#fdfdfd"   # 10.00+
]
precip_colormap = matplotlib.colors.ListedColormap(nws_precip_colors)


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
                                                  vmin=0, vmax=30,
                                                  cmap=precip_colormap)
    def init():
        im.set_array(np.array([]))
        return im,

    def animate(i):
        data = da.isel(time=i)
        time = pd.to_datetime(data.time.values)
        title = str(data.name)+' - '+time.strftime('%Y-%m-%d %H')
        im.set_array(data.values.ravel())
        ax.set_title(title)
        return (im,)
    
    return animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(da.coords['time']), interval=100, 
                                   blit=True)

anim = create_animation(da)

HTML(anim.to_html5_video())

anim.save('tp_2005_hourly_nomask.mp4')

HTML(anim.to_jshtml())

