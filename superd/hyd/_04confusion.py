"""

evaluating coarse wsh grids (vs. fine)
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

 




 

def write_confusion_stack(coarse_nc_dir=None,
                           coarse_fp_l=None,
                           fine_fp=None,
                           out_dir=None,
                             encoding = {'zlib': True, 'complevel': 5, 'dtype': 'int16'},
                          max_workers=5,
 
                           ):
    """compute performance stats for extent and value"""
    
    
    #===========================================================================
    # defautls
    #===========================================================================
 
    if out_dir is None:
        out_dir=os.path.join(lib_dir, '04_confu')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name='perf')
    
    #===========================================================================
    # load the fine
    #===========================================================================
    #da_fine =  xr.open_dataarray(fine_fp, engine='netcdf4', mask_and_scale=True,chunks={'x':-1, 'y':-1, 'tag':1})
    
    #===========================================================================
    # get filepaths
    #===========================================================================\
    if coarse_fp_l is None:
        log.info(f'building from {coarse_nc_dir}')
        coarse_fp_l = get_filepaths(coarse_nc_dir, ext='.nc')
    
    
    log.info(f'w/ {len(coarse_fp_l)} coarse files\n' + '\n    '.join([os.path.basename(fp) for fp in coarse_fp_l]))
    log.info(f'from fine {os.path.basename(fine_fp)}')
    
    #===========================================================================
    # load the cooarse
    #===========================================================================
    """not concating the coarse and fine as they have different shaped indexes (fine is missing Mannings)"""
    
    with xr.open_mfdataset(coarse_fp_l, 
                           #parallel=True,  
                       engine='netcdf4',
                       data_vars='minimal',
                       combine="nested",
                       concat_dim='tag',
                       decode_coords="all",
                       chunks={'x':-1, 'y':-1, 'tag':1, 'MannningsValue':1},
                   ) as ds_coarse, xr.open_dataarray(fine_fp, engine='netcdf4', chunks={'x':-1, 'y':-1, 'tag':1}
                                        ) as da_fine:
     
        log.info(f'loaded {ds_coarse.dims}'+
             f'\n    coors: {list(ds_coarse.coords)}'+
             f'\n    data_vars: {list(ds_coarse.data_vars)}'+
             f'\n    crs:{ds_coarse.rio.crs}'
             )
        
        
        
        #re-order to match tag coords in other dataArray
        da_coarse = ds_coarse.loc[{'tag':da_fine['tag'].values}]['wd_max']
 
        
        #check
        for k in ['x', 'y', 'tag']:
            assert np.array_equal(da_coarse.coords[k].values, da_fine.coords[k].values), k
    

     
        #===========================================================================
        # data prep
        #===========================================================================
     
        
        #convert to boolean
        
        da_coarseB = xr.where(da_coarse.fillna(0.0)>0.0, True, False) #wet:True, dry:False
        
     
        da_fineB = xr.where(da_fine.fillna(0.0)>0.0, True, False)
        
     
        #===========================================================================
        # calc for each tag
        #===========================================================================
        keys = ['tag', 'MannningsValue']
        ofp_lib  = dict()
        
        #===========================================================================
        # single core
        #===========================================================================
        if max_workers is None:
            for gkey0, gda_fineB in da_fineB.groupby(keys[0], squeeze=True):
         
                gda_coarseB = da_coarseB.loc[{'tag':gkey0}] #get this coarse
         
                log.info(f'computing for {gkey0}')
                
                #=======================================================================
                # calc for each mannings
                #=======================================================================
                   
                ofp_lib[gkey0] =  _confu_loop(gda_fineB, gda_coarseB, out_dir, encoding, keys, gkey0,
                                              log=log.getChild(gkey0))
                
     
                
        #===========================================================================
        # multi-core
        #===========================================================================
        else:
            log.info(f'w/ max_workers={max_workers}')
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_multi_confu_loop, gkey0, gda_fineB, da_coarseB, out_dir, encoding, keys
                                    ) for gkey0, gda_fineB in da_fineB.groupby(keys[0], squeeze=True)]
                for future in concurrent.futures.as_completed(futures):
                    gkey0 = future.result()
                    ofp_lib[gkey0] = future.result()
            
     
            
    #=======================================================================
    # wrap
    #=======================================================================
    log.info(f'finishedto \n    {out_dir}')
    
    return out_dir
        
 
        


def _multi_confu_loop(gkey0, gda_fineB, da_coarseB, out_dir, encoding, keys):
    
 
    return _confu_loop(gda_fineB, da_coarseB.loc[{'tag':gkey0}], out_dir, encoding, keys, gkey0)

def _confu_loop(gda_fineB, gda_coarseB,out_dir, encoding, keys, gkey0, log=None):
    
    if log is None:
        log = get_log_stream()
        
    ofp_d = dict()
     
    log.info(f'building confusion mats for \'{gkey0}\' on {len(gda_coarseB.MannningsValue)}')
    for gkey1, gdaB_i in gda_coarseB.groupby(keys[1], squeeze=True):
    #setup
        keys_d = dict(zip(keys, [gkey0, gkey1]))
        uuid = hashlib.md5(f'{keys_d}'.encode("utf-8")).hexdigest()
        ofp_d[gkey1] = os.path.join(out_dir, f'confu_{gkey0}_{gkey1}_{uuid}.nc')
        #cnt += 1
    #write
        if not os.path.exists(ofp_d[gkey1]):
            log.info(f'    get_confusion_cat for {keys_d} to')
            conf_ar = get_confusion_cat(gda_fineB.squeeze().values, gdaB_i.values, confusion_codes=confusion_codes)
            #write
            log.info(f'        to_netcdf to {ofp_d[gkey1]}')
            conf_da = xr.DataArray(data=conf_ar, coords=gdaB_i.coords, name='confusion').astype(int)
            conf_da.encoding.update(encoding)
            conf_da.to_netcdf(ofp_d[gkey1], mode='w', format='netcdf4', engine='netcdf4', compute=True)
        else:
            log.debug(f'already exists {keys_d}... skipping')
        log.debug(f'finished {keys_d}')
    
    return ofp_d

 

    
if __name__=="__main__":
 
    kwargs = dict(
        coarse_nc_dir=r'l:\10_IO\2307_super\lib\03_resample',
        fine_fp=r'l:\10_IO\2307_super\lib\02_clip\concat_clip_fine_5-1688-5224_20230807.nc', 
            )
    
    write_confusion_stack(**kwargs)
    
    

    
    
    
    
    
    
    
    
    
    