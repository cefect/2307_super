"""

evaluating coarse coarse wsh grids (vs. fine)
"""


import os, hashlib, logging, shutil, psutil, webbrowser
from datetime import datetime


import numpy as np
import pandas as pd

#import dask
import concurrent.futures


from osgeo import gdal # Import gdal before rasterio

from rasterio.enums import Resampling

import rioxarray
import xarray as xr

 

from definitions import lib_dir, wrk_dir, temp_dir

from superd.hyd.ahr_params import epsg_id, scenarioTags_d

from superd.hyd.coms import load_nc_to_xarray, confusion_codes

from superd.hp import (
    init_log, today_str, get_filepaths, dstr, get_confusion_cat,
    get_log_stream,
    )

 
def compute_inundation_performance(
                confu_dir=None,
                confu_fp_l=None,
                
                out_dir=None,
                # encoding = {'zlib': True, 'complevel': 5, 'dtype': 'int16'},
                # max_workers=5,
 
                           ):
    """compute performance on inundation using confusion grids"""
    
    #===========================================================================
    # defautls
    #===========================================================================
 
    if out_dir is None:
        out_dir=os.path.join(wrk_dir, 'performance', 'inundation', today_str)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name='perf')
 
    
    #===========================================================================
    # get filepaths
    #===========================================================================\
    if confu_fp_l is None:
        log.info(f'building from {confu_dir}')
        confu_fp_l = get_filepaths(confu_dir, ext='.nc', nested=True)
    
    
    log.info(f'w/ {len(confu_fp_l)}  files\n')
 
    
    #===========================================================================
    # load the cooarse
    #===========================================================================
    """not concating the coarse and fine as they have different shaped indexes (fine is missing Mannings)"""
    
    #===========================================================================
    # with xr.open_mfdataset(confu_fp_l, 
    #                        #parallel=True,  
    #                    engine='netcdf4',
    #                    data_vars='minimal',
    #                    combine="nested",
    #                    concat_dim=['tag', 'ManningsValue'],
    #                    decode_coords="all",
    #                    compat='broadcast_equals',
    #                    coords='minimal',
    #                    #chunks={'x':-1, 'y':-1, 'tag':1, 'MannningsValue':1},
    #                ) as ds_raw:
    #===========================================================================
    
    for i, fp_l in enumerate(confu_fp_l):
        log.info(f'{i}  on {len(fp_l)}')
        
 
        #combine a list of raster netcdfs, each with identical x,y and a single unique ManningsValue
        with xr.open_mfdataset(fp_l,  
                   engine='netcdf4',                   
                   combine="nested",        
                   concat_dim='ManningsValue',
                   #coords =['ManningsValue'], 
                   #chunks={'x':-1, 'y':-1, 'MannningsValue':1},
               ) as ds_raw:
         
            log.info(f'loaded {ds_raw.dims}'+
                 f'\n    coors: {list(ds_raw.coords)}'+
                 f'\n    data_vars: {list(ds_raw.data_vars)}'+
                 f'\n    crs:{ds_raw.rio.crs}'
                 )
            
            #prep
            ds_raw['ManningsValue']
            ds_raw.coords['ManningsValue'].values
            ds_raw.assign_coords('ManningsValue')
            da = ds_raw['confusion']
            
            #=======================================================================
            # compute standard confusion metrics
            #=======================================================================
            
            for (gkey0, gkey1), gda in da.groupby(['tag', 'ManningsValue']):
                log.info(f'on {gkey0}.{gkey1}')
        
        
        
 
        
 
            
     
            
    #=======================================================================
    # wrap
    #=======================================================================
    log.info(f'finishedto \n    {out_dir}')
    
    return out_dir
        
 
        
 

 

    
if __name__=="__main__":
 
    kwargs = dict(
        confu_dir=r'l:\10_IO\2307_super\lib\04_confu2',  
            )
    
    compute_inundation_performance(**kwargs)
    
    

    
    
    
    
    
    
    
    
    
    
