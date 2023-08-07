'''
Created on Aug. 6, 2023

@author: cefect


load asc files into  xarray dataset
    impute metadata from filenames
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


def _get_asc_meta(scenarioTags_d, fp):
    """get meta on a single asc"""
    fn = os.path.basename(fp)
#get scenario
    tag = None
    for k, v in scenarioTags_d.items():
        if v in fn:
            tag = k
    
    assert not tag is None
    d = {'tag':tag, 'fn':fn}
        #data_var
    for k1 in ['wd_max']:
        if k1 in fn:
            d['data_var'] = k1
    
    assert 'data_var' in d, f'failed to match data_var'
#filesize
    d['filesize_MB'] = os.path.getsize(fp) / (1024 ** 2)
    d['resolution'] = get_meta(fp)['res'][0]
    return d

def _build_metadf(fp_l, scenarioTags_d, lookup_df):
    """build coarse meta frame"""
    key_lib = dict()
    for i, fp in enumerate(fp_l):
        d = _get_asc_meta(scenarioTags_d, fp)
 
 
        #mannings
        coln = 'MannningsValue'
        fn = os.path.basename(fp)
        try:
            k = int(fn.split('_')[2])
        except Exception as e:
            raise KeyError(f'failed to extract {coln} w/ \n    {e}')
        d[coln] = lookup_df[coln][k]
 
        #wrap
        key_lib[i] = d
    
    meta_df = pd.DataFrame.from_dict(key_lib).T
    return meta_df


def _get_asc_files(search_dir, count_file_limiter, log):
    fp_l_raw = get_filepaths(search_dir, ext='.asc')
    log.info(f'identified {len(fp_l_raw)} files')
    if not count_file_limiter is None:
        log.warning(f'only taking first {count_file_limiter}')
        fp_l = fp_l_raw[:count_file_limiter]
        
        assert len(fp_l)==count_file_limiter
    else:
        fp_l = fp_l_raw
    return fp_l

def load_coarse_asc_concat_all(
        search_dir=None,
        out_dir=None,
        count_file_limiter=None,
        scenarioTags_d=scenarioTags_d,
        lookup_fp=r'l:\10_IO\2307_super\ins\hyd\ahr\20230804\ManningsFor32m.txt',
        meta_pick_fp=None,
        client=None,
        encoding = {
            #'zlib': True, 'complevel': 5, 
            'dtype': 'float32', 'least_significant_digit':2},
        ):
    
    """load a set of asc results files and use to populate a sparase data array
    
    spent some time trying to get the lock to work
        seems like using threaded raises lock issues
        maybe parallelize check operations on the inner loop?
    
    Pars
    -----------
    lookup_fp: str
        filepath to ta table that converts indexers to mannigns values
        
    """
    
    #===========================================================================
    # defaults
    #===========================================================================
    start = datetime.now() 
    
    assert os.path.exists(search_dir), search_dir    
    
    if out_dir is None:
        out_dir=os.path.join(lib_dir, '01_concat')
    if not os.path.exists(out_dir):os.makedirs(out_dir)
 
    #===========================================================================
    # logger
    #===========================================================================
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name='load')
    log.info(f'from {search_dir}\n\n')
    
    
    #===========================================================================
    # build meta-------
    #===========================================================================
    if meta_pick_fp is None:
        #===========================================================================
        # file search
        #===========================================================================
        fp_l = _get_asc_files(search_dir, count_file_limiter, log)
            
        #===========================================================================
        # build meta
        #===========================================================================
        #looad lookup table
        lookup_df = pd.read_csv(lookup_fp,sep=' ', index_col=0)
        log.info(f'loaded lookup_df={str(lookup_df.shape)} from \n    {lookup_fp}')
        
        #build the metadata
        log.info(f'building metadata from {len(fp_l)}')
        meta_df = _build_metadf(fp_l, scenarioTags_d, lookup_df)
        log.info(f'built meta_df {str(meta_df.shape)}')
        
        #=======================================================================
        # write pick
        #=======================================================================
        meta_pick_fp = os.path.join(out_dir, f'meta_{os.path.basename(search_dir)}_{len(meta_df)}')
        meta_df.to_pickle(meta_pick_fp+'.pkl')
        meta_df.to_csv(meta_pick_fp+'.csv')
 
        
        log.info(f'wrote meta_df to \n    {meta_pick_fp}.pkl')
        
    else:
        log.warning(f'loading meta from \n    {meta_pick_fp}')

        meta_df = pd.read_pickle(meta_pick_fp)
 
    #===========================================================================
    # load-----------
    #===========================================================================
    log.info(f'looping and loading on {len(meta_df)}')
    
    
    #===========================================================================
    # loop per group
    #===========================================================================
    da_dtag = dict()
    
    #with Lock("rio-read", client=client) as lock: #create a lock from the client
    #lock = Lock("rio-read", client=client)
    for gval, gdf in meta_df.groupby('tag'):
        da_d=dict()
        
        log.info(f'building for {gval}={len(gdf)}')
        for i, row in gdf.iterrows():
            d = row.to_dict()
     
            fp = os.path.join(search_dir, row['fn'])
     
            
            da_i = rioxarray.open_rasterio(fp,
                                    parse_coordinates =True,
                                    chunks =True, #load lazily w/ auto chunks
                                    cache=False,
                                    masked =True,
                                    #lock=lock,
                                    lock=False,
                                    default_name = d.pop('data_var'),
                                    )
            
            log.debug(f'    {gval}.{i+1}/{len(meta_df)} loadded {da_i.shape} w/ {da_i.nbytes}')
            
            #do some checks
            """not sure how to do this in parallel
            if __debug__:
                log.info(f'_check_wsh_da')
                if not _check_wsh_da(da_i, log, i ,d):
                    continue"""
         
            
            #add some meta
            da_i = da_i.assign_coords({k:d.pop(k) for k in ['tag', 'MannningsValue']})
            da_i.attrs.update(d)
            
            #wrap
            da_d[i] = da_i.squeeze()
        
        log.info(f'finished loading {gval} in {(datetime.now() - start).total_seconds():.2f} secs. coconcat ')
        
        #===========================================================================
        # concat the tag  
        #===========================================================================
        da_dtag[gval] = xr.concat(da_d.values(), dim='MannningsValue', combine_attrs='drop')
            
    #===========================================================================
    # concat all
    #===========================================================================
    log.info(f'concat on {len(da_dtag)} tags')
    da_m = xr.concat(da_dtag.values(), dim='tag', combine_attrs='drop')
    
    #set compression
    da_m.encoding.update(encoding)
    
    #fill nulls
    da_m = da_m.where(da_m!=0)
    
    #===========================================================================
    # do some checks
    #===========================================================================
     
    #===========================================================================
    # write------
    #===========================================================================
    shape_str = '-'.join([str(e) for e in da_m.shape])
    ofp = os.path.join(out_dir, f'concat_coarse_{shape_str}_{today_str}.nc')
    
    log.info(f'to_netcdf')
    #with Lock("rio-write", client=client) as lock:
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






def _check_wsh_da(da_i, log, i, d):
    valid = True
    cnt = np.sum(da_i < 0.0).compute()
    if cnt > 0:
        log.error(f'{i} got {cnt} vals <0...skipping\n    {d}')
        valid= False
    #continue
    maxv = da_i.max().compute()
    if maxv > 99.0:
        log.error(f'{i} got max ({maxv})>99  ...skipping\n    {d}')
        valid= False
    #continue
    del cnt, maxv
    return valid
    
 
def load_fine_asc_concat_all(
        search_dir=None,
        out_dir=None,
        count_file_limiter=None,
        scenarioTags_d=scenarioTags_d,
  
        meta_pick_fp=None,
        client=None,
        encoding = {
            #'zlib': True, 'complevel': 5, 
            'dtype': 'float32', 'least_significant_digit':2},
        ):
     
    """load a set of asc results files and use to populate a sparase data array
     
    Pars
    -----------
    lookup_fp: str
        filepath to ta table that converts indexers to mannigns values
         
    """
     
    #===========================================================================
    # defaults
    #===========================================================================
    start = datetime.now() 
     
    assert os.path.exists(search_dir), search_dir    
     
    if out_dir is None:
        out_dir=os.path.join(lib_dir, '01_concatF')
    if not os.path.exists(out_dir):os.makedirs(out_dir)
  
    #===========================================================================
    # logger
    #===========================================================================
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name='loadF')
    log.info(f'from {search_dir}\n\n')
     
    #===========================================================================
    # build meta--------
    #===========================================================================
    if meta_pick_fp is None:
        #===========================================================================
        # file search
        #===========================================================================
        fp_l = _get_asc_files(search_dir, count_file_limiter, log)
         
        #=======================================================================
        # build meta
        #======================================================================= 
        key_lib = {i:_get_asc_meta(scenarioTags_d, fp) for i, fp in enumerate(fp_l)}
         
        meta_df = pd.DataFrame.from_dict(key_lib).T        
        log.info(f'built meta_df {str(meta_df.shape)}')        
        #=======================================================================
        # write pick
        #=======================================================================
        meta_pick_fp = os.path.join(out_dir, f'metaF_{os.path.basename(search_dir)}_{len(meta_df)}')
        meta_df.to_pickle(meta_pick_fp+'.pkl')
        meta_df.to_csv(meta_pick_fp+'.csv') 
         
        log.info(f'wrote meta_df to \n    {meta_pick_fp}')
         
         
    else:
        log.warning(f'loading meta from \n    {meta_pick_fp}')
 
        meta_df = pd.read_pickle(meta_pick_fp)
         
         
    #===========================================================================
    # load-----------
    #===========================================================================
    log.info(f'looping and loading on {len(meta_df)}')
     
     
    #===========================================================================
    # loop per group
    #===========================================================================
    da_d = dict()
     
    #setup lock
    #===========================================================================
    # if client is None:
    #     lock=None
    # else:
    #     lock = Lock("rio-read", client=client) #create a lock from the client
    #===========================================================================
  
    for i, row in meta_df.iterrows():
        d = row.to_dict()
  
        fp = os.path.join(search_dir, row['fn']) 
         
        da_i = rioxarray.open_rasterio(fp,
                                parse_coordinates =True,
                                chunks =True, #load lazily w/ auto chunks
                                cache=False,
                                masked =True,
                                #lock=lock,
                                lock=False,
                                default_name = d.pop('data_var'),
                                )
         
        log.info(f'     {i+1}/{len(meta_df)} loadded {da_i.shape} w/ {da_i.nbytes}')
         
        #do some checks
        if __debug__:
            if not _check_wsh_da(da_i, log, i ,d):
                continue
         
      
         
        #add some meta
        da_i = da_i.assign_coords({k:d.pop(k) for k in ['tag']})
        da_i.attrs.update(d)
         
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



 
    
if __name__=="__main__":
    
    pass
    
 
    
    #===========================================================================
    # coarse
    #===========================================================================
 
 #==============================================================================
 #    kwargs = dict(        
 #        
 #            #search_dir=r'l:\10_IO\2307_super\ins\hyd\ahr\20230804\004m\raw',
 #            search_dir=r'l:\10_IO\2307_super\ins\hyd\ahr\20230804\032m\raw',
 #            #meta_pick_fp=r'l:\10_IO\2307_super\lib\01_concat\meta_raw_100.pkl',
 #            meta_pick_fp=r'l:\10_IO\2307_super\lib\01_concat\meta_raw_1494.pkl',
 # 
 #            #count_file_limiter=100,
 #            #processes=True, #this works
 #            #processes=False, n_workers=None, #fails
 #            #processes=False, n_workers=1, #lock fail
 #            processes=True, threads_per_worker=1, #works, throws RuntimeWarning: coroutine
 #            )
 # 
 # 
 #    dask_run_cluster(load_coarse_asc_concat_all, **kwargs)
 #==============================================================================
    
    
    #===========================================================================
    # fine
    #===========================================================================
    
 #==============================================================================
 #    kwargs = dict(        
 # 
 #            search_dir=r'l:\10_IO\2307_super\ins\hyd\ahr\20230804\004m\raw',
 # 
 #            #meta_pick_fp=r'l:\10_IO\2307_super\lib\01_concat\meta_raw_1494.pkl',
 # 
 # 
 #            processes=True, threads_per_worker=1, #works, throws RuntimeWarning: coroutine
 #            )
 #    
 #    dask_run_cluster(load_fine_asc_concat_all, **kwargs)
 #==============================================================================
        

    
    
    
    
    
    
    

