for q in [0.25, .5, .75]:
    print('percentile', q, ': ', round(float(dis_mean.quantile(q)),3), 'm^3/s')

dist = dis_mean.values.ravel()
sns.distplot(dist, bins=np.logspace(-1, 4), hist_kws=dict(cumulative=True))
plt.xlim(0,100)
plt.grid()
plt.ylabel('cumulative distribution')
plt.xlabel('discharge [m$^3$/s]')

from python.aux.utils_floodmodel import cluster_by_discharge
from python.aux.utils import calc_area, nandot

bin_edges = [0, 0.8, 2.5, 10.25, 10000]
cluster = cluster_by_discharge(dis_mean, bin_edges)

for c in cluster:
    plt.figure()
    cluster[c].plot()
    plt.title('#points: '+str(int(cluster[c].sum())))

image = dis_mean*0.
image.name = 'spatial feature cluster'
for i, c in enumerate(cluster):
    image = image.where(~cluster[c], i)
    
image.plot(cmap = mpl.colors.ListedColormap(['grey', 'orange', 'blue', 'darkblue']))

cluster = cluster.to_array('clusterId')
cluster.coords

def aggregate_clustersum(ds, cluster, clusterdim):
    """Aggregate a 3-dimensional array over certain points (latitude, longitude).

    Parameters
    ----------
    ds : xr.Dataset
        the array to aggregate (collapse) spatially
    cluster : xr.DataArray
        3-dimensional array (clusterdim, latitude, longitude),
        `clusterdim` contains the True/False mask of points to aggregate over
        e.g. len(clusterdim)=4 means you have 4 clusters
    clusterdim : str
        dimension name to access the different True/False masks

    Returns
    -------
    xr.DataArray
        1-dimensional
    """
    out = xr.Dataset()

    # enforce same coordinates
    interp = True
    if (len(ds.latitude.values) == len(cluster.latitude.values) and
            len(ds.longitude.values) == len(cluster.longitude.values)):
        if (np.allclose(ds.latitude.values, cluster.latitude.values) and
                np.allclose(ds.longitude.values, cluster.longitude.values)):
            interp = False
    if interp:
        ds = ds.interp(latitude=cluster.latitude, longitude=cluster.longitude)
    area_per_gridpoint = calc_area(ds.isel(time=0))

    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()

    for var in ds:
        for cl in cluster.coords[clusterdim]:
            newname = var+'_cluster'+str(cl.values)
            this_cluster = cluster.sel({clusterdim: cl})

            da = ds[var].where(this_cluster, 0.)  # no contribution from outside cluster
            #print(da)
            out[newname] = xr.dot(da, area_per_gridpoint)
    return out.drop(clusterdim)

# later, we can import the function from here
# from python.aux.utils_flowmodel import aggregate_clustersum

Xagg = aggregate_clustersum(X, cluster, 'clusterId')

# drop these predictors
for v in Xagg:
    if 'cluster0' in v:
        for vn in ['lsp-5-11', 'lsp-12-25', 'lsp-26-55', 'lsp-56-180']:
            if vn in v:
                Xagg = Xagg.drop(v)
                break

# drop these predictors (predictand time)
for v in Xagg:
    for vn in ['lsp_cluster', 'cp_cluster']:
        if v.startswith(vn):
            Xagg = Xagg.drop(v)
            break

if False:  # alternative: aggregating over space by taking the mean
    Xagg = X.mean(['latitude', 'longitude'])

Xagg

