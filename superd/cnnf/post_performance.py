'''
Created on Dec. 10, 2023

@author: cef

compute performance of flood grids
'''
import os, psutil
from datetime import datetime
import numpy as np

from osgeo import gdal # Import gdal before rasterio
import rioxarray
import xarray as xr

import geopandas as gpd
import fiona
import shapely.geometry
from pyproj import CRS


from definitions import wrk_dir
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
            out_dir = os.path.join(wrk_dir, 'outs', 'SRCNN-flood', 'eval')
     
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        ofp = os.path.join(out_dir, f'pred_{today_str}.tif')
        
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
        #da_i = da_i.assign_coords({k:d.pop(k) for k in ['tag']})
        da_i.attrs.update({'scenario':scen})
         
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
    shape_str = '-'.join([str(e) for e in da_m.shape])
    ofp = os.path.join(out_dir, f'concat_fine_{shape_str}_{today_str}.nc')
     
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
        
    