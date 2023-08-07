"""

evaluating coarse wsh grids (vs. fine)
"""


import os, hashlib, logging, shutil, psutil, webbrowser
from datetime import datetime


import numpy as np
import pandas as pd

import dask


from osgeo import gdal # Import gdal before rasterio
import rioxarray
import xarray as xr

import matplotlib.pyplot as plt
import seaborn as sns

from definitions import lib_dir, wrk_dir

from superd.hyd.ahr_params import epsg_id, scenarioTags_d

from superd.hp import init_log, today_str, get_filepaths, dstr


from superd.hp import init_log, today_str, get_filepaths
from superd.hyd.coms import load_nc_to_xarray






def calc_performance_stats(coarse_nc_fp=None,
                           fine_nc_fp=None,
                           out_dir=None,
                           pick_d=dict(),
                           ):
    """compute performance stats for extent and value"""
    
    
    #===========================================================================
    # defautls
    #===========================================================================
    #nc filepaths
    def get_nc_fp(fp, subdir):
        if fp is None:
            fp =  get_filepaths(os.path.join(lib_dir, subdir), count=1)
            
        assert os.path.exists(fp), fp
        return fp        
     
    coarse_nc_fp =get_nc_fp(coarse_nc_fp, '01_concat')    
    fine_nc_fp =get_nc_fp(fine_nc_fp, '01_concatF')
        
    
    if out_dir is None:
        out_dir=os.path.join(wrk_dir, 'performance', today_str)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name='perf')
    
    log.info(f'building from \n    coarse: {coarse_nc_fp}\n    fine: {fine_nc_fp}')
    #===========================================================================
    # CSI
    #===========================================================================
    if not 'csi' in pick_d:
        get_csi_fromStack(coarse_nc_fp, fine_nc_fp, log=log)
        
        


    
def get_csi_fromStack(coarse_nc_fp, fine_nc_fp, log=None):
    """calc CSI of coarse against fine"""
    
    
    #===========================================================================
    # load arras
    #===========================================================================
    daF =  xr.open_dataarray(fine_nc_fp, engine='netcdf4', mask_and_scale=True, 
        chunks={'x':-1, 'y':-1, 'tag':1})
        
    log.info(f'loaded FINE {daF.shape} w/ \n    {daF.dims}\n    {daF.chunks}')
    
    
    daC = xr.open_dataarray(coarse_nc_fp, engine='netcdf4', mask_and_scale=True, 
        chunks={'x':-1, 'y':-1, 'tag':1, 'MannningsValue':1})
        
    log.info(f'loaded COARSE {daC.shape} w/ \n    {daC.dims}\n    {daC.chunks}')
    
    #group on daC and calc each
    
    
    
    
    
if __name__=="__main__":
 
    kwargs = dict(
            fine_nc_fp=r'l:\10_IO\2307_super\lib\01_concatF\concat_5-1688-5230_20230807.nc',
            coarse_nc_fp= r'l:\10_IO\2307_super\lib\01_concat\concat_5-299-211-654_20230806.nc',      
 
            )
    
    calc_performance_stats(**kwargs)
    
    

    
    
    
    
    
    
    
    
    
    