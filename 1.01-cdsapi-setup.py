#!pip install cdsapi

UID = '139134'
API_key = '57bdd8fa-fda5-4037-a52c-23871113c263'

import os
with open(os.path.join(os.path.expanduser('~'), '.cdsapirc'), 'w') as f:
    f.write('url: https://cds.climate.copernicus.eu/api/v2\n')
    f.write(f'key: {UID}:{API_key}')

# Import cdsapi and create a Client instance
import cdsapi
c = cdsapi.Client()
# More complex request
c.retrieve("reanalysis-era5-pressure-levels", {
            "product_type":   "reanalysis",
            "format":         "netcdf",
            "area":           "52.00/2.00/40.00/20.00", # N/W/S/E
            "variable":       "geopotential",
            "pressure_level": "500",
            "year":           "2017",
            "month":          "01",
            "day":            "12",
            "time":           "00"
            }, "example_era5_geopot_700.nc")

