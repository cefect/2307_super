'''
Created on Aug. 23, 2023

@author: cefect

misc tools for working with the raster sets
'''


#===============================================================================
# IMPORTS----------
#===============================================================================
import os, hashlib, logging, shutil, psutil, webbrowser, sys
from datetime import datetime

#import threading

import numpy as np
import pandas as pd

from osgeo import gdal # Import gdal before rasterio
import rioxarray
import xarray as xr

from definitions import lib_dir, wrk_dir
from superd.hyd.ahr_params import epsg_id, scenarioTags_d, coln_d

from superd.hyd.hp import init_log, today_str, get_filepaths, get_meta, dstr


def write_geotiffs_from_coords(tag_man_d,
                               nc_fp=None,
                              out_dir=None,
 
                              ):
    """lookup a simulation result and write as a GeoTiff
    
    usefull for extracting the optimum
    
    Pars
    --------
    tag_man_d: dict
        tag: mannings value
        
    nc_fp: str
        filepath to output of 'clip' routine (single dataarray)
    """
    
    #===========================================================================
    # defaults
    #===========================================================================
    start = datetime.now() 
    
  
    
    if out_dir is None:
        out_dir=os.path.join(wrk_dir,'outs', 'raster_tools', today_str)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name='rtools')
    
    log.info(f'extracting from : {nc_fp} ')
    
    #===========================================================================
    # load the array
    #===========================================================================
    ofp_d = dict()
    with xr.open_dataarray(nc_fp, engine='netcdf4', mask_and_scale=True, 
        chunks={'x':-1, 'y':-1, 'tag':1, coln_d['man']:1}
        ).rio.write_crs(f'EPSG:{epsg_id}'
        ).rio.write_nodata(-9999 
                      ) as da:
        
        log.info(f'loaded {da.shape} w/ \n    {da.dims}\n    {da.chunks}\n    res={da.rio.resolution()}')
        
        for i, (tag, n) in enumerate(tag_man_d.items()):
            log.info(f'    extracting {tag}.{n}')
            
            #select
            """not sure why this doesn't work...
            da[{'tag':tag, coln_d['man']:n}]"""
            
            da_i = da.sel(MannningsValue=n, tag=tag)
            
            #write
            ofp_d[i] = os.path.join(out_dir, f'{da_i.name}_{tag}_n{n}_{today_str}.tif')
            log.info(f'    writing to: {ofp_d[i]}')
            da_i.fillna(0.0).rio.to_raster(ofp_d[i],
                                                  dtype='float32',
                                                  recalc_transform =True,                                                  
                                                  )
            
    #===========================================================================
    # wrap
    #===========================================================================
    log.info(f'finished on {len(ofp_d)}\n{dstr(ofp_d)}')
    
    return ofp_d
 
        
    
    
    
    
    

if __name__=="__main__":
    
    kwargs=dict(  
        tag_man_d = {
            'base':0.084,
            'n10':0.078,
            'n20':0.095,
            'p10':0.085,
            'p20':0.096,
            },
        nc_fp=r'l:\10_IO\2307_super\lib\02_clip\concat_clip_coarse_5-299-211-653_20230807.nc',
        )
        
    write_geotiffs_from_coords(**kwargs)
    
 