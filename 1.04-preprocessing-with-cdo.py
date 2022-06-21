from cdo import Cdo
import glob
import xarray as xr
cdo = Cdo()
tmp_file = './tmp.nc'

xar = xr.open_mfdataset(glob.glob(path_to_data+'era5_precipitation*.nc'),
                        combine='by_coords')
xar.to_netcdf(tmp_file)

cdo.daysum(input=tmp_file,
           output=path_to_data+'era5_precip_daysum.nc')
os.remove(tmp_file)

import sys
sys.path.append('../')

path_to_data = 'volume/project/data/'

from ml_flood.python.aux.utils import cdo_daily_means

incl = 'temperature'

cdo_daily_means(path=path_to_data, file_includes=incl)

from ml_flood.python.aux.utils import cdo_precip_sums

incl = 'large_scale_precipitation'

cdo_precip_sums(path=path_to_data, file_includes=incl)

from ml_flood.python.aux.utils import cdo_clean_precip

cdo_clean_precip(path=path_to_data, precip_type='precipitation')

from ml_flood.python.aux.utils import cdo_spatial_cut

lonmin = 10
lonmax = 20
latmin = 40
latmax = 50
incl = 'temperature'
incl_new = 'temperature_spatial_cut'

cdo_spatial_cut(path=path_to_data, file_includes=incl, new_file_includes=incl_new, lonmin=lonmin, lonmax=lonmax, latmin=latmin, latmax=latmax)

from ml_flood.python.aux.utils import cdo_merge_time

incl = 'temperature'
new_filename = 'temperature_YYYYinit-YYYYend.nc'

cdo_merge_time(path=path_to_data, file_includes=incl, new_file=new_filename)

