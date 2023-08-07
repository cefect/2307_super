'''
Created on Aug. 7, 2023

@author: cefect


building resample of coarse array

'''



import os, hashlib, logging, shutil, psutil, webbrowser, gc
from datetime import datetime


import numpy as np
import pandas as pd

import dask


from osgeo import gdal # Import gdal before rasterio

from rasterio.enums import Resampling

import rioxarray
import xarray as xr
 

from definitions import lib_dir, wrk_dir, temp_dir

from superd.hyd.ahr_params import epsg_id, scenarioTags_d

from superd.hyd.coms import load_nc_to_xarray

from superd.hp import init_log, today_str, get_filepaths, dstr, dask_run_cluster, dask_run_threads


    
def write_coarse_resample(coarse_nc_fp, 
                          new_shape=(1688, 5224), 
                          out_dir=None,
                          client=None,
                          encoding = {'zlib': True, 'complevel': 5},
                          ):
    """resample coarse and write"""
    
    #===========================================================================
    # defautls
    #===========================================================================
    start = datetime.now()
    #nc filepaths
    def get_nc_fp(fp, subdir):
        if fp is None:
            fp =  get_filepaths(os.path.join(lib_dir, subdir), count=1)
            
        assert os.path.exists(fp), fp
        return fp        
     
    coarse_nc_fp =get_nc_fp(coarse_nc_fp, '02_clip')    
 
        
    
    if out_dir is None:
        out_dir=os.path.join(wrk_dir, lib_dir, '03_resample')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name='resamp')
    
    log.info(f'building from \n    coarse: {coarse_nc_fp} ')
    
    
    #===========================================================================
    # load arras
    #===========================================================================
    #===========================================================================
    # daF =  xr.open_dataarray(fine_nc_fp, engine='netcdf4', mask_and_scale=True, 
    #     chunks={'x':-1, 'y':-1, 'tag':1}).rio.write_crs(f'EPSG:{epsg_id}')
    #     
    # log.info(f'loaded FINE {daF.shape} w/ \n    {daF.dims}\n    {daF.chunks}\n    res={daF.rio.resolution()}')
    #===========================================================================
    
    
    daC = xr.open_dataarray(coarse_nc_fp, engine='netcdf4', mask_and_scale=True, 
        chunks={'x':-1, 'y':-1, 'tag':1, 'MannningsValue':1}).rio.write_crs(f'EPSG:{epsg_id}')
        
    log.info(f'loaded COARSE {daC.shape} w/ \n    {daC.dims}\n    {daC.chunks}\n    res={daC.rio.resolution()}')
    
    #===========================================================================
    # check shapes
    #===========================================================================
    #===========================================================================
    # d=dict() 
    # for i in [1, 2]: 
    #     d[i] = daF.shape[i]/daC.shape[i+1]
    #     assert int(d[i])==float(d[i]), f' dim {i} not event\n    %s'%d
    #===========================================================================
        
    #===========================================================================
    # build coarse resample
    #===========================================================================
        
 
    
    #group on daC and calc each
    #===========================================================================
    # calc for each tag
    #===========================================================================
    #new_shape = (daF.shape[1], daF.shape[2])
    
 
    ofp_d  = dict()
    for gkey, gdaC in daC.groupby('tag'):
        #log.info(f'computing for {gkey}')

        #defaults
        uuid = hashlib.md5(f'{gkey}_{new_shape}_{coarse_nc_fp}'.encode("utf-8")).hexdigest()
        ofp_d[gkey] = os.path.join(out_dir, f'coarse_reproject_{gkey}_{uuid}.nc')
        
        if not os.path.exists(ofp_d[gkey]):
            #=======================================================================
            # resmplae
            #=======================================================================
            log.info(f'resampling \'{gkey}\' {gdaC.shape} to {new_shape}')
            
            """loads to memory.. need to write to release"""
            gdaC_f = gdaC.rio.reproject(daC.rio.crs,
                               shape=new_shape,
                               resampling=Resampling.nearest,
                               ).expand_dims(dim='tag').reset_encoding()
            
            log.info(f'resampled {gkey} from {gdaC.shape} to {gdaC_f.shape} w/ res={gdaC_f.rio.resolution()}. writing')
            
            #add promote dimensionless 'tag' back to a dimensional coordinate 
            gdaC_f.encoding.update(encoding)
            #=======================================================================
            # write
            #=======================================================================
    
            gdaC_f.to_netcdf(ofp_d[gkey], mode ='w', format ='netcdf4', engine='netcdf4', compute=True)
            log.info(f'wrote {gdaC_f.shape} to \n    %s'%ofp_d[gkey])
            
            gdaC_f.close()
            del gdaC_f
            gc.collect()
            
            
            
        else:
            log.info(f'{gkey} exists... skipping')
        
 #==============================================================================
 #    #===========================================================================
 #    # collect
 #    #===========================================================================
 #    log.info(f'finished writing to {len(ofp_d)}\n    {dstr(ofp_d)}')
 #    
 #    with xr.open_mfdataset(list(ofp_d.values()), parallel=True,  
 #                           engine='netcdf4',
 #                           data_vars='minimal',
 #                           combine="nested",
 #                           concat_dim='tag',
 #                           decode_coords="all",
 #                           chunks={'x':-1, 'y':-1, 'tag':1, 'MannningsValue':1},
 #                       ) as ds:
 #    
 #        log.info(f'loaded {ds.dims}'+
 #             f'\n    coors: {list(ds.coords)}'+
 #             f'\n    data_vars: {list(ds.data_vars)}'+
 #             f'\n    crs:{ds.rio.crs}'
 #             )
 #        
 #        #add some meta
 #        ds = ds.assign_attrs({'resolution':4})
 # 
 #        shape_str = '-'.join([str(e) for e in ds['wd_max'].shape])
 #        ofp = os.path.join(out_dir, f'coarse_resamp_{shape_str}_{today_str}.nc')
 #        
 #        log.info(f'writing {shape_str} to \n    {ofp}')
 #        ds.to_netcdf(ofp, mode ='w', format ='netcdf4', engine='netcdf4', compute=True)
 #==============================================================================
        
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
 
    kwargs = dict(
            #fine_nc_fp=r'l:\10_IO\2307_super\lib\02_clip\concat_clip_fine_5-1688-5224_20230807.nc',
            coarse_nc_fp= r'l:\10_IO\2307_super\lib\02_clip\concat_clip_coarse_5-299-211-653_20230807.nc',
            new_shape=(1688, 5224),      
 
            )
    
    write_coarse_resample(**kwargs)
    
    #dask_run_cluster(write_coarse_resample, **kwargs, processes=True, n_workers=8, threads_per_worker=1)
    
    #dask_run_threads(write_coarse_resample, **kwargs)
    