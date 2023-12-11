'''
Created on Aug. 6, 2023

@author: cefect

heavy helper for superd.hyd
    2023-12-11: this use to have all hp functions,but now split out to better manage dependneices
'''

import os, logging, pprint, sys 

from datetime import datetime

import numpy as np
import pandas as pd



from dask.distributed import LocalCluster as Cluster
from dask.distributed import Client
from dask.diagnostics import ResourceProfiler, visualize
import dask
from dask.diagnostics import ProgressBar

#import xarray as xr

 

from hp.logr import *
from hp.basic import *
from hp.rio import *
#===============================================================================
# logging
#===============================================================================
 
#supress some warnings
from bokeh.util.warnings import BokehUserWarning
import warnings
warnings.simplefilter(action='ignore', category=BokehUserWarning)
    


#===============================================================================
# DASK--------
#===============================================================================

def dask_run_cluster(func, *args, 
                     n_workers=None,
                     threads_per_worker=None, 
                     memory_limit='auto', 
                     processes=True,
                     **kwargs):
        #start a cluster and connect client
    with Cluster( 
                      threads_per_worker=threads_per_worker, 
                    n_workers=n_workers,
                     memory_limit=memory_limit, 
                     processes=processes,
                       ) as cluster, Client(cluster) as client:
        
        print(f' opening dask client {client.dashboard_link}')
        webbrowser.open(client.dashboard_link)
        
        with ResourceProfiler(dt=0.25) as rprof: 
            func(*args, client=client, **kwargs)
    
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
    
 

def dask_run_threads(func, n_workers=None, **kwargs):
 
    
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
    



# 

 
