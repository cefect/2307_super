'''
Created on Aug. 6, 2023

@author: cefect


load asc files into  xarray dataset
    impute metadata from filenames
'''


#===============================================================================
# IMPORTS----------
#===============================================================================
import os, hashlib, logging, shutil, psutil, webbrowser
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

 

from definitions import epsg_id, lib_dir

from superd.hp import init_log, today_str, get_filepaths

 

def _build_metadf(fp_l, scenarioTags_d, lookup_df):
    key_lib = dict()
    for i, fp in enumerate(fp_l):
        fn = os.path.basename(fp)
        
        #get scenario
        tag = None
        for k, v in scenarioTags_d.items():
            if v in fn:
                tag = k
        
        assert not tag is None
        d = {'tag':tag, 'fn':fn}
 
        #mannings
        coln = 'MannningsValue'
        try:
            k = int(fn.split('_')[2])
        except Exception as e:
            raise KeyError(f'failed to extract {coln} w/ \n    {e}')
        d[coln] = lookup_df[coln][k]
        #data_var
        for k1 in ['wd_max']:
            if k1 in fn:
                d['data_var'] = k1
        
        assert 'data_var' in d, f'failed to match data_var'
        #filesize
        d['filesize_MB'] = os.path.getsize(fp) / (1024 ** 2)
        #d['fp'] = fp #long and ugly... and not portable
        #wrap
        key_lib[i] = d
    
    meta_df = pd.DataFrame.from_dict(key_lib).T
    return meta_df

def load_asc_concat_all(
        search_dir=None,
        out_dir=None,
        count_file_limiter=None,
        scenarioTags_d={
            'n10':'Minus10cm',
            'n20':'Minus20cm',
            'base':'OG',
            'p10':'Plus10cm',
            'p20':'Plus20cm'            
            },
        lookup_fp=r'l:\10_IO\2307_super\ins\hyd\ahr\20230804\ManningsFor32m.txt',
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
        out_dir=os.path.join(lib_dir, '01_concat')
    if not os.path.exists(out_dir):os.makedirs(out_dir)
 
    #===========================================================================
    # logger
    #===========================================================================
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name='load')
    log.info(f'from {search_dir}\n\n')
    
    
    if meta_pick_fp is None:
        #===========================================================================
        # file search
        #===========================================================================
        fp_l_raw = get_filepaths(search_dir, ext='.asc')
        log.info(f'identified {len(fp_l_raw)} files')
        
     
        if not count_file_limiter is None:
            log.warning(f'only taking first {count_file_limiter}')
            fp_l = fp_l_raw[:count_file_limiter]
        else:
            fp_l = fp_l_raw
            
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
    da_dtag = dict()
    
    lock = Lock("rio-read", client=client) #create a lock from the client
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
                                    lock=lock,
                                    default_name = d.pop('data_var'),
                                    )
            
            log.debug(f'    {gval}.{i+1}/{len(meta_df)} loadded {da_i.shape} w/ {da_i.nbytes}')
         
            
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
    # write
    #===========================================================================
    shape_str = '-'.join([str(e) for e in da_m.shape])
    ofp = os.path.join(out_dir, f'concat_{shape_str}_{today_str}.nc')
    
    log.info(f'to_netcdf')
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
    
        
def run_with_dask(**kwargs):
        #start a cluster and connect client
    with LocalCluster( 
                       #========================================================
                       #  threads_per_worker=13, 
                       n_workers=1,
                       # memory_limit='auto', 
                       # processes=False,
                       #========================================================
                       ) as cluster, Client(cluster) as client:
        
        print(f' opening dask client {client.dashboard_link}')
        webbrowser.open(client.dashboard_link)
    
        return load_asc_concat_all(client=client, **kwargs)

 
    
if __name__=="__main__":
 
    kwargs = dict(
        
        
            search_dir=r'l:\10_IO\2307_super\ins\hyd\ahr\20230804\032m\raw',
            #meta_pick_fp=r'l:\10_IO\2307_super\lib\01_concat\meta_raw_1494.pkl',
 
            count_file_limiter=None,
            )
    run_with_dask(**kwargs)
    
    
    #load_asc_concat_all(**kwargs)
        
        
        


    
    
    
    
    
    
    

