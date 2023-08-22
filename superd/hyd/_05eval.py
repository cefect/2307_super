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

from superd.hyd.coms import confusion_codes, coln_d

from superd.hp import (
    init_log, today_str, get_filepaths, dstr, get_confusion_cat,
    get_log_stream,get_directory_size
    )

from superd.hyd.fperf import ValidateMask

 
def compute_inundation_performance(
                confu_dir=None,
                confu_fp_l=None,
                
                out_dir=None,
                # encoding = {'zlib': True, 'complevel': 5, 'dtype': 'int16'},
                # max_workers=5,
 
                           ):
    """compute performance on inundation using confusion grids"""
    start = datetime.now()
    #===========================================================================
    # defautls
    #===========================================================================
 
    if out_dir is None:
        out_dir=os.path.join(wrk_dir, 'performance', 'inundation', today_str)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name='iperf')
 
    
    #===========================================================================
    # get filepaths
    #===========================================================================\
    if confu_fp_l is None:
        log.info(f'building from {confu_dir}')
        confu_fp_l = get_filepaths(confu_dir, ext='.nc', nested=True)
        
 
    
    
    log.info(f'w/ {len(confu_fp_l)}  files\n')
 
    #===========================================================================
    # load nested confusion
    #===========================================================================
    """not concating the coarse and fine as they have different shaped indexes (fine is missing Mannings)"""
 
    ds_d = dict()
    for i, fp_l in enumerate(confu_fp_l):
        #fp_l = fp_l[:10]
        
        log.info(f'{i+1}/{len(confu_fp_l)}  on {len(fp_l)}')
        
        
 
        #combine a list of raster netcdfs, each with identical x,y and a single unique ManningsValue
        with xr.open_mfdataset(fp_l,
                                #fp_l[:3],  
                               parallel=True,
                   engine='netcdf4',                   
                   combine="nested",        
                   concat_dim=coln_d['man'],
                   #coords =['ManningsValue'], 
                   chunks={'x':-1, 'y':-1, coln_d['man']:1},
               ) as ds_raw:
         
            #===================================================================
            # log.info(f'loaded {ds_raw.dims}'+
            #      f'\n    coors: {list(ds_raw.coords)}'+
            #      f'\n    data_vars: {list(ds_raw.data_vars)}'+
            #      f'\n    crs:{ds_raw.rio.crs}'
            #      )
            #===================================================================
            
            ds_d[i] = ds_raw
            
    #concat
    log.info(f'concat on {len(ds_d)}')
    ds = xr.concat(ds_d.values(), dim='tag')
    
    log.info(f'loaded {ds.dims}'+
         f'\n    coors: {list(ds.coords)}'+
         f'\n    data_vars: {list(ds.data_vars)}'+
         f'\n    crs:{ds.rio.crs}'
         )
    
    da = ds['confusion']
    
    """
    ds['confusion'].isel(MannningsValue=0, tag=0).values
    """
            
 
    #=======================================================================
    # compute standard confusion metrics
    #=======================================================================
    res_d =dict()
    for gkey0, gda_i in da.groupby('tag'):
        d = dict()
        for gkey1, gda in gda_i.groupby(coln_d['man']):
            log.info(f'ValidateMask on {gkey0}.{gkey1}')
            
            try:
                with ValidateMask(confu_ar=gda.values, logger=log) as wrkr:
                    d[gkey1] = wrkr.get_inundation_all()
            except Exception as e:
                log.warning(f'failed on {gkey0}.{gkey1} w/ \n    {e}')
                
        #wrap gkey1
        res_d[gkey0] = pd.DataFrame.from_dict(d).T
        res_d[gkey0].index.name = coln_d['man']
        
    #wrap gkey0
    res_dx = pd.concat(res_d, names=['tag'])
    
    #===========================================================================
    # wrap
    #===========================================================================
    log.info(f'finished w/ {res_dx.shape}')
    
    #write
    shape_str = '-'.join([str(e) for e in list(res_dx.shape)])
    ofp = os.path.join(out_dir, f'eval_inun_metrics_{shape_str}_{today_str}')
    
    res_dx.to_pickle(ofp+'.pkl')
    res_dx.to_csv(ofp+'.csv')
    
    log.info(f'wrote to \n    {ofp}')
    
    #===========================================================================
    # wrap
    #===========================================================================
    meta_d = {
                    'tdelta':(datetime.now()-start).total_seconds(),
                    'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    'disk_GB':get_directory_size(out_dir),
                    #'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
    log.info(meta_d)
 
    
    return out_dir
        
 
        
 

 

    
if __name__=="__main__":
 
    kwargs = dict(
        confu_dir=r'l:\10_IO\2307_super\lib\04_confu2',  
            )
    
    compute_inundation_performance(**kwargs)
    
    dask_run_cluster
    
    

    
    
    
    
    
    
    
    
    
    
