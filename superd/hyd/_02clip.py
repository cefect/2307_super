'''
Created on Aug. 7, 2023

@author: cefect
'''


#===============================================================================
# IMPORTS----------
#===============================================================================
import os, hashlib, logging, shutil, psutil, webbrowser, sys
from datetime import datetime

import threading

import numpy as np
import pandas as pd

import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster, Lock
from dask.utils import SerializableLock

from osgeo import gdal # Import gdal before rasterio
import rioxarray
import xarray as xr

 

from definitions import lib_dir

from superd.hyd.ahr_params import epsg_id, scenarioTags_d

from superd.hp import init_log, today_str, get_filepaths, get_meta, dask_run_cluster



def clip_dataArray(fp, 
                   new_shape,
                   scenario='fine',
                   out_dir=None,
                   ):
    """clip a data array
    
    doing this when the shapes are not evenly divisible"""
    
    #===========================================================================
    # defaults
    #===========================================================================
    start = datetime.now() 
    if out_dir is None:
        out_dir=os.path.join(lib_dir, '02_clip')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name='perf')
    
    log.info(f'clipping to {new_shape} on \n    {fp}')
    
    #===========================================================================
    # open and clip
    #===========================================================================
    with xr.open_dataarray(fp, engine='netcdf4', mask_and_scale=True, 
        chunks='auto') as da:
        
        log.info(f'loaded {scenario} with {da.shape} w/ \n    {da.dims}\n    {da.chunks}')
        
        #build indexer
        if scenario=='fine':
            assert da.dims==('tag','y', 'x')
            #=======================================================================
            # da_i_x = xr.DataArray(da.coords['x'].values[:new_shape[1]], dims=["x"])
            # da_i_y = xr.DataArray(da.coords['y'].values[:new_shape[2]], dims=["y"])
            #=======================================================================
            da_i_y = xr.DataArray(np.arange(new_shape[1]), dims=["y"])
            da_i_x = xr.DataArray(np.arange(new_shape[2]), dims=["x"])
            
            
            #make slice
            da_s = da[:, da_i_y, da_i_x]
            
        else:
            assert da.dims==('tag', 'MannningsValue', 'y', 'x')
            da_i_x = xr.DataArray(np.arange(new_shape[3]), dims=["x"])
            da_i_y = xr.DataArray(np.arange(new_shape[2]), dims=["y"])
            
            #make slice
            da_s = da[:, :,  da_i_y, da_i_x]
        
        assert da_s.shape==new_shape
        
 
        #===========================================================================
        # write
        #===========================================================================
        shape_str = '-'.join([str(e) for e in da_s.shape])
        ofp = os.path.join(out_dir, f'concat_clip_{scenario}_{shape_str}_{today_str}.nc')
         
        log.info(f'to_netcdf')
        #with Lock("rio", client=client) as lock:
        da_s.to_netcdf(ofp, mode ='w', format ='netcdf4', engine='netcdf4', compute=True)
         
        log.info(f'merged all w/ {da_s.shape} and wrote to \n    {ofp}')
         
      
        #===========================================================================
        # wrap
        #===========================================================================
        meta_d = {
                        'tdelta':(datetime.now()-start).total_seconds(),
                        'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                        #'disk_GB':get_directory_size(temp_dir),
                        'output_MB':os.path.getsize(ofp)/(1024**2)
                        }
        log.info(meta_d)
         
        return ofp

if __name__=="__main__":
    #===========================================================================
    # clip
    #===========================================================================
    clip_dataArray(r'l:\10_IO\2307_super\lib\01_concatF\concat_fine_5-1688-5230_20230807.nc', (5, 1688, 653*8), scenario='fine')
    #clip_dataArray(r'l:\10_IO\2307_super\lib\01_concat\concat_5-299-211-654_20230807.nc', (5, 299, 211, 653), scenario='coarse')
    
     
         
