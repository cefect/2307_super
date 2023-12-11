'''
Created on Dec. 10, 2023

@author: cef

compute performance of flood grids
'''
import os, psutil
from datetime import datetime
import numpy as np
import pandas as pd

from osgeo import gdal # Import gdal before rasterio
import rioxarray
import xarray as xr

import geopandas as gpd
import fiona
import shapely.geometry
from pyproj import CRS


from definitions import wrk_dir
from parameters import confusion_codes

from hp.basic import today_str
from hp.logr import get_log_stream

from superd.fperf import get_confusion_cat
#from superd.hyd._01asc_to_concat import _check_wsh_da


def gtif_to_xarray(
        fp_d,
        aoi_fp=None,
        out_dir=None,log=None,ofp=None,
        encoding = {
            #'zlib': True, 'complevel': 5, 
            'dtype': 'float32', 'least_significant_digit':2},
        
        ):
    """load a list of GeoTiffs into a rio.xarray"""
    
    #===========================================================================
    # setup
    #===========================================================================
    start = datetime.now() 
    #configure outputs
    if ofp is  None:
        if out_dir is None:
            out_dir = os.path.join(wrk_dir, 'outs', 'cnnf', 'gtif_to_xarray')
     
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        ofp = os.path.join(out_dir, f'wsh_concat_xr_{today_str}.nc')
        
    if log is None: log  = get_log_stream('gtif_to_xarray') #get the root logger
    
    log.info(f'on {len(fp_d)}')
    
    #===========================================================================
    # bounding box
    #===========================================================================
    if aoi_fp:
 
        with fiona.open(aoi_fp, "r") as source:
            bbox = shapely.geometry.box(*source.bounds) 
            crs = CRS(source.crs['init'])
        #=======================================================================
        # with fiona.open(aoi_fp, 'r') as geojson:
        #     polygons = [feature for feature in geojson]
        #     assert len(polygons)==1
        #     polygons[0]['geometry']['coordinates'].bounds
        #     poly = shapely.geometry.mapping(polygons[0])
        #=======================================================================
            
        
    
    #===========================================================================
    # loop and load
    #===========================================================================
    da_d=dict()
    for i, (scen, fp) in enumerate(fp_d.items()):
        da_i = rioxarray.open_rasterio(fp,
                                parse_coordinates =True,
                                chunks =True, #load lazily w/ auto chunks
                                cache=False,
                                masked =True,
                                #lock=lock,
                                lock=False,
                                default_name = 'WSH',
                                )
         
        log.info(f'     \'{scen}\' loaded {da_i.shape}')
         
        #do some checks
        if __debug__:
            cnt = np.sum(da_i < 0.0).compute().item()
            assert cnt==0, f'{scen} got {cnt} values <0.0'
            
            maxv = da_i.max().compute().item()
            assert maxv<99.0, f'{scen} got a maximum value of {maxv}'
            
            
 
        #=======================================================================
        # clip
        #=======================================================================
        if aoi_fp:
            da_i = da_i.rio.clip_box(*bbox.bounds)
      
         
        #add some meta
        da_i = da_i.assign_coords({'tag':scen})
        da_i.attrs.update({'fn':os.path.basename(fp)})
         
        #wrap
        da_d[i] = da_i.squeeze()
     
    log.info(f'finished loading {i} in {(datetime.now() - start).total_seconds():.2f} secs. coconcat ')
 
    #===========================================================================
    # concat the tag  
    #===========================================================================
    log.info(f'concat on {len(da_d)} tags')
    da_m = xr.concat(da_d.values(), dim='tag', combine_attrs='drop')
         
    #===========================================================================
    # post
    #===========================================================================
    #set compression
    da_m.encoding.update(encoding)
     
    #fill nulls
    da_m = da_m.where(da_m!=0)
     
 
     
    #===========================================================================
    # write
    #===========================================================================
 
     
    log.info(f'to_netcdf')
    #with Lock("rio", client=client) as lock:
    da_m.to_netcdf(ofp, mode ='w', format ='netcdf4', engine='netcdf4', compute=True)
     
    log.info(f'merged all w/ {da_m.shape} and wrote to \n    {ofp}')
     
  
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


def build_confusion_xr(
        nc_fp, true_tag='hyd_fine',
        out_dir=None, ofp=None, log=None,
        encoding=dict(),
        ):
    """build confusion grids for each WSH in the stack"""
    
    #===========================================================================
    # setup
    #===========================================================================
    start = datetime.now() 
    #configure outputs
    if ofp is  None:
        if out_dir is None:
            out_dir = os.path.join(wrk_dir, 'outs', 'cnnf', 'build_confusion_xr')
     
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        ofp = os.path.join(out_dir, f'confusion_xr_{today_str}.nc')
        
    if log is None: log  = get_log_stream('build_confusion_xr') #get the root logger
    
    log.info(f'on {nc_fp}')
    
    #===========================================================================
    # open data array
    #===========================================================================
    with xr.open_dataarray(nc_fp, engine='netcdf4', chunks={'x':-1, 'y':-1, 'tag':1}
                                        ) as da:
        log.info(f'loaded {da.dims}'+
             f'\n    coors: {list(da.coords)}'+
            # f'\n    data_vars: {list(da.data_vars)}'+
             f'\n    crs:{da.rio.crs}'
             )
        
        #=======================================================================
        # convert to boolean
        #=======================================================================
        wetB_da = xr.where(da>0.0, True, False) #wet:True, dry:False
        
        #=======================================================================
        # loop and compute
        #=======================================================================
        #get the true
        true_wetB_ar = wetB_da.loc[{'tag':true_tag}].squeeze().values
        
        #loop on each predicted
        d = dict()
        for tag, gdaB_i in wetB_da.groupby('tag', squeeze=False):
            if tag==true_tag: continue
            
            log.info(f'computing confusion array on {tag}')
            #compute the confusion array
            conf_ar = get_confusion_cat(true_wetB_ar, 
                                        gdaB_i.values, 
                                        confusion_codes=confusion_codes)
            
            d[tag] = xr.DataArray(data=conf_ar, coords=gdaB_i.coords, name='confusion').astype(int)
            
    #=======================================================================
    # combine
    #=======================================================================
    #join datasets together on a new axis
    confu_da = xr.concat(d.values(), pd.Index(d.keys(), name='tag', dtype=str))

    #set compression
    confu_da.encoding.update(encoding)
     
 
     
    #===========================================================================
    # write
    #===========================================================================
 
     
    log.info(f'to_netcdf')
    #with Lock("rio", client=client) as lock:
    confu_da.to_netcdf(ofp, mode ='w', format ='netcdf4', engine='netcdf4', compute=True)
     
    log.info(f'merged all w/ {confu_da.shape} and wrote to \n    {ofp}')
    
    return ofp
        
        
def compute_inundation_performance(nc_fp,
                
                out_dir=None,log=None, ofp=None,
                # encoding = {'zlib': True, 'complevel': 5, 'dtype': 'int16'},
                # max_workers=5,
 
                           ):
    """compute performance on inundation using confusion grids
 
    """
    
    raise IOError('stopped here')
    #===========================================================================
    # setup
    #===========================================================================
    start = datetime.now() 
    #configure outputs
    if ofp is  None:
        if out_dir is None:
            out_dir = os.path.join(wrk_dir, 'outs', 'cnnf', 'compute_inundation_performance')
     
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        ofp = os.path.join(out_dir, f'confusion_{today_str}.nc')
        
    if log is None: log  = get_log_stream('build_confusion_xr') #get the root logger
    
    log.info(f'on {nc_fp}')
 
 
 
    #===========================================================================
    # open data array
    #===========================================================================
    with xr.open_dataarray(nc_fp, engine='netcdf4', chunks={'x':-1, 'y':-1, 'tag':1}
                                        ) as da:
        log.info(f'loaded {da.dims}'+
             f'\n    coors: {list(da.coords)}'+
            # f'\n    data_vars: {list(da.data_vars)}'+
             f'\n    crs:{da.rio.crs}'
             )
            
    
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
        
 
        
 
        
         


        
    