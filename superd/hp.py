'''
Created on Aug. 6, 2023

@author: cefect
'''

import os, logging, pprint, webbrowser, sys
import logging.config
from datetime import datetime
import pandas as pd
from definitions import wrk_dir, logcfg_file

from dask.distributed import LocalCluster as Cluster
from dask.distributed import Client
from dask.diagnostics import ResourceProfiler, visualize
import dask
from dask.diagnostics import ProgressBar

import xarray as xr

#===============================================================================
# logging
#===============================================================================
 
#supress some warnings
from bokeh.util.warnings import BokehUserWarning
import warnings
warnings.simplefilter(action='ignore', category=BokehUserWarning)
    
today_str = datetime.now().strftime('%Y%m%d')

#===============================================================================
# loggers-----------
#===============================================================================
log_format_str = '%(levelname)s.%(name)s.%(asctime)s:  %(message)s'
def init_root_logger(
 
        log_dir = wrk_dir,
        ):
    logger = logging.getLogger() #get the root logger
    
    logging.config.fileConfig(logcfg_file,
                              defaults={'logdir':str(log_dir).replace('\\','/')},
                              disable_existing_loggers=True,
                              ) #load the configuration file 
    
    logger.info(f'root logger initiated and configured from file: {logcfg_file}\n    logdir={log_dir}')
    
    return logger

def get_new_file_logger(
        name='r',
        level=logging.DEBUG,
        fp=None, #file location to log to
        logger=None,
        ):
    
    #===========================================================================
    # configure the logger
    #===========================================================================
    if logger is None:
        logger = logging.getLogger(name)
        
    logger.setLevel(level)
    
    #===========================================================================
    # configure the handler
    #===========================================================================
    assert fp.endswith('.log')
    
    formatter = logging.Formatter(log_format_str)        
    handler = logging.FileHandler(fp, mode='w') #Create a file handler at the passed filename 
    handler.setFormatter(formatter) #attach teh formater object
    handler.setLevel(level) #set the level of the handler
    
    logger.addHandler(handler) #attach teh handler to the logger
    
    logger.info('built new file logger  here \n    %s'%(fp))
    
    return logger
 
def init_log(
 
        log_dir=wrk_dir,
        **kwargs):
    """wrapper to setup the root loger and create a file logger"""
    
    root_logger = init_root_logger(log_dir=log_dir) 
    
    #set up the file logger
    return get_new_file_logger(**kwargs)


def get_log_stream(name=None, level=None):
    """get a logger with stream handler"""
    if name is None: name=str(os.getpid())
    if level is None:
        if __debug__:
            level=logging.DEBUG
        else:
            level=logging.INFO
    
    logger = logging.getLogger(name)
    
    #see if it has been configured
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(
            stream=sys.stdout, #send to stdout (supports colors)
            ) #Create a file handler at the passed filename 
        formatter = logging.Formatter(log_format_str) 
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
    
#===============================================================================
# MISC-----------
#===============================================================================
def dstr(d,
         width=100, indent=0.3, compact=True, sort_dicts =False,
         ):
    return pprint.pformat(d, width=width, indent=indent, compact=compact, sort_dicts =sort_dicts)

def view(df):
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)
    import webbrowser
    #import pandas as pd
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(delete=False, suffix='.html', mode='w') as f:
        #type(f)
        df.to_html(buf=f)
        
    webbrowser.open(f.name)
    
#===============================================================================
# files/folders---------
#===============================================================================

def get_temp_dir(temp_dir_sfx = r'py\temp'):
    
    from pathlib import Path
    homedir = str(Path.home())
    temp_dir = os.path.join(homedir, temp_dir_sfx)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    return temp_dir


def get_directory_size(directory):
    total_size = 0
    for path, dirs, files in os.walk(directory):
        for f in files:
            fp = os.path.join(path, f)
            total_size += os.path.getsize(fp)
    return total_size / 1024 / 1024 / 1024

def get_filepaths(search_dir, ext='.nc', count=None):
    """search a directory for files with the provided extension. return the first result"""
    assert os.path.exists(search_dir), search_dir
    find_l = list()
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.endswith(ext):
                find_l.append(os.path.join(root, file))
    
    if not count is None:
        assert len(find_l)==count,f'failed to get uinique match {len(find_l)} from \n    {search_dir}'
        if count ==1:
            return find_l[0]
        else:
            raise NotImplementedError(count)
    else:
        return find_l


 

#===============================================================================
# dask runners--------
#===============================================================================
def dask_profile_func(func, *args,
                      threads_per_worker=16, n_workers=1,
                      **kwargs):
    
    with Client(threads_per_worker=threads_per_worker, n_workers=n_workers,
                 memory_limit='auto', processes=False,
                 ) as client:
          
        print(f' opening dask client {client.dashboard_link}')
        webbrowser.open(client.dashboard_link)
          
        with ResourceProfiler(dt=0.25) as rprof: 
            func(*args, **kwargs)
 
             
        #profile results
        _wrap_rprof(rprof) 

        """seems to be ignoring the filename kwarg"""
        """this also doesn't fix it
        os.chdir(os.path.expanduser('~'))"""
        
        rprof.visualize(
            #filename=os.path.join(os.path.expanduser('~'), f'dask_ReserouceProfile_{today_str}.html'),
            #filename=os.path.join(wrk_dir, f'dask_ReserouceProfile_{today_str}.html')
            )
        
    return rprof

def dask_threads_func(func, n_workers=None, **kwargs):
 
    
    pbar = ProgressBar(5.0, dt=1.0) #show progress bar for computatinos greater than 5 secs
    pbar.register()  
  
    #zarr_rechunker(**kwargs)
    with dask.config.set(scheduler='threads', n_workers=n_workers):
        func(**kwargs)

def _wrap_rprof(rprof):
 
    # initialize variables to store the maximum values
    max_mem = 0
    max_cpu = 0
    # iterate over the results to find the maximum values
    for result in rprof.results:
        max_mem = max(max_mem, result.mem)
        max_cpu = max(max_cpu, result.cpu)
    
    total_time = rprof.results[-1].time - rprof.results[0].time
    # print the maximum values
    print(f"total_time={total_time:.2f} secs, max_mem={max_mem:.2f} MB, max_cpu={max_cpu:.1f} %")
    
    
#===============================================================================
# SPARSE------
#===============================================================================
def dataArray_todense(da):
    """convert a sparse dataArray to a dense one"""

    return xr.DataArray(
        da.data.todense(),
        dims=da.dims, coords=da.coords
        )
    

def write_sparse_xarray(
        ds_sparse, ofp, log=None,sparse_datavar=None,sparse_index=None
        ):
    """
    write an xarray Dataset with sparse data as two files
    
    (some data, y, x)
    """
    #===========================================================================
    # defaults
    #===========================================================================
    if log is None: log=logging.getLogger('write') 
        
    #ofp = ofp + '_' + '-'.join([str(abs(v)) for v in coo_ar.shape])
    
    if sparse_datavar is None: sparse_datavar=ds_sparse.attrs['sparse_datavar']
    
    #get the shape of the data source
    dm = ds_sparse.squeeze().dims
   
    
    if sparse_index is None:
        sparse_index = list(set(dm.keys()).difference(['x', 'y']))[0]
    
 

    #dshape = [dm[sparse_index], dm['y'], dm['x']]
    
    #===========================================================================
    # check
    #===========================================================================
    assert np.array_equal(
            np.arange(0, len(ds_sparse.coords[sparse_index])),
            ds_sparse.coords[sparse_index].values), f'got discontinous sparse index'
 
    
    #===========================================================================
    # write sparse
    #===========================================================================
    ofp1 = ofp+'.npz'
    
    sparse.save_npz(ofp1, ds_sparse[sparse_datavar].data, compressed=True)
    
    #===========================================================================
    # write xarray
    #===========================================================================
    ds_empty = ds_sparse.drop(sparse_datavar) 
 
    ds_empty.assign_attrs({'sparse_filename':ofp1, 'sparse_datavar':sparse_datavar, 'sparse_index':sparse_index}
                    ).to_netcdf(ofp+'.nc', mode ='w', format ='netcdf4', engine='netcdf4', compute=True)
    
    log.info(f'wrote { ds_sparse[sparse_datavar].shape} to \n    {os.path.dirname(ofp)}')
    
    return ofp1


