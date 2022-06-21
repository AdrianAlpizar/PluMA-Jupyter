import sys
sys.path.append('../')
from ml_flood.python.misc.data_download import CDS_Dataset

# define name of dataset to request
ds = CDS_Dataset(dataset_name='reanalysis-era5-pressure-levels',
                 save_to_folder='./example_download/'  # path to where datasets shall be stored
                )

# define areas of interest
area_dict = dict(danube=[50, 7, 47, 20],
                 asia=[55, -140, 0, 35],
                 usa=[50, -125, 25, -70])

# define time frame
year_start = 2000
year_end = 2000
month_start = 1
month_end = 12

# define requested variables
request = dict(product_type='reanalysis', 
               format='netcdf',
               area=area_dict['usa'],
               variable=['geopotential', 'temperature', 'specific_humidity'], 
               pressure_level=['850', '700', '500'])

# start request of data (1 year worth)
ds.get(years = [str(y) for y in range(year_start, year_end+1)],
       months = [str(a).zfill(2) for a in range(month_start, month_end+1)],
       request = request,
       N_parallel_requests = 12)

