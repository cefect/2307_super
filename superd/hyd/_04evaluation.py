"""

evaluating coarse wsh grids (vs. fine)
"""


import os, hashlib, logging, shutil, psutil, webbrowser
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

from superd.hp import init_log, today_str, get_filepaths, dstr, get_confusion_cat

 




 

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
        chunks={'x':-1, 'y':-1, 'tag':1}).rio.write_crs(f'EPSG:{epsg_id}')
        
    log.info(f'loaded FINE {daF.shape} w/ \n    {daF.dims}\n    {daF.chunks}\n    res={daF.rio.resolution()}')
    
    
    daC = xr.open_dataarray(coarse_nc_fp, engine='netcdf4', mask_and_scale=True, 
        chunks={'x':-1, 'y':-1, 'tag':1, 'MannningsValue':1}).rio.write_crs(f'EPSG:{epsg_id}')
        
    log.info(f'loaded COARSE {daC.shape} w/ \n    {daC.dims}\n    {daC.chunks}\n    res={daC.rio.resolution()}')
    
    #===========================================================================
    # check shapes
    #===========================================================================
    d=dict() 
    for i in [1, 2]: 
        d[i] = daF.shape[i]/daC.shape[i+1]
        assert int(d[i])==float(d[i]), f' dim {i} not event\n    %s'%d
        
    #===========================================================================
    # build coarse resample
    #===========================================================================
        
 
    
    #group on daC and calc each
    #===========================================================================
    # calc for each tag
    #===========================================================================
    new_shape = (daF.shape[1], daF.shape[2])
    
 
    ofp_d  = dict()
    for gkey, gdaC in daC.groupby('tag'):
        #log.info(f'computing for {gkey}')

        #defaults
        uuid = hashlib.md5(f'{gkey}_{new_shape}_{coarse_nc_fp}'.encode("utf-8")).hexdigest()
        ofp_d[gkey] = os.path.join(temp_dir, f'coarse_reproject_{gkey}_{uuid}.nc')
        
        if not os.path.exists(ofp_d[gkey]):
            #=======================================================================
            # resmplae
            #=======================================================================
            log.info(f'resampling \'{gkey}\' {gdaC.shape} to {new_shape}')
            
            """loads to memory.. need to write to release"""
            gdaC_f = gdaC.rio.reproject(daF.rio.crs,
                               shape=new_shape,
                               resampling=Resampling.nearest,
                               )
            
            log.info(f'resampled {gkey} from {gdaC.shape} to {gdaC_f.shape} w/ res={gdaC_f.rio.resolution()}. writing')
            
            #add meta
            
            #=======================================================================
            # write
            #=======================================================================
    
            gdaC_f.to_netcdf(ofp_d[gkey], mode ='w', format ='netcdf4', engine='netcdf4', compute=True)
            log.info(f'wrote {gdaC_f.shape} to \n    %s'%ofp_d[gkey])
            
            gdaC_f.close()
            
        else:
            log.info(f'{gkey} exists... skipping')
        
    #===========================================================================
    # collect
    #===========================================================================
    log.info(f'finished writing to {len(ofp_d)}')
        
    
    #===========================================================================
    # calc for each tag
    #===========================================================================
    for gkey, gdaC in daC.groupby('tag'):
        log.info(f'computing for {gkey}')
        
        #get fine
        daF_i = daF.loc[{'tag':gkey}]
        
        
        """
        daF_i.plot.imshow()
        """
        #=======================================================================
        # inundation performance----
        #=======================================================================
        #=======================================================================
        # convert to boolean
        #=======================================================================
        log.info(f'converting to boolean')
        daF_i_bool = daF_i.fillna(0.0).where(daF_i>=0, True, False).astype(bool) #0:dry, 1:wet
        gdaC_i_f_bool = gdaC_f.fillna(0.0).where(gdaC_f>=0, True, False).astype(bool)  #0:dry, 1:wet
        
        #=======================================================================
        # confusion
        #=======================================================================
        log.info(f'building confusion mat on {len(gdaC_i_f_bool.MannningsValue)}')
        for gkey1, gdaC_i_f_bool_j in gdaC_i_f_bool.groupby('MannningsValue'):
            get_confusion_cat(daF_i_bool, gdaC_i_f_bool)
 
        
 

    
if __name__=="__main__":
 
    kwargs = dict(
            fine_nc_fp=r'l:\10_IO\2307_super\lib\02_clip\concat_clip_fine_5-1688-5224_20230807.nc',
            coarse_nc_fp= r'l:\10_IO\2307_super\lib\02_clip\concat_clip_coarse_5-299-211-653_20230807.nc',      
 
            )
    
    calc_performance_stats(**kwargs)
    
    

    
    
    
    
    
    
    
    
    
    