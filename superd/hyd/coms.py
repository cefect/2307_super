'''
Created on Aug. 6, 2023

@author: cefect
'''
import os, logging, pprint, webbrowser, sys
import pandas as pd


from osgeo import gdal # Import gdal before rasterio
import rioxarray
import xarray as xr


confusion_codes={'TP':11, 'TN':12, 'FP':21, 'FN':22}


def load_nc_to_xarray(fp,
                      ):
    """load a netcdf file to xarray"""
    
    assert os.path.exists(fp), fp
    
