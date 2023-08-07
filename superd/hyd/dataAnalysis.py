'''
Created on Aug. 6, 2023

@author: cefect

'''

#===============================================================================
# PLOT ENV------
#===============================================================================

#===============================================================================
# setup matplotlib----------
#===============================================================================
env_type = 'draft'
cm = 1 / 2.54

if env_type == 'journal': 
    usetex = True
elif env_type == 'draft':
    usetex = False
elif env_type == 'present':
    usetex = False
else:
    raise KeyError(env_type)

 
 
  
import matplotlib
#matplotlib.use('Qt5Agg') #sets the backend (case sensitive)
matplotlib.set_loglevel("info") #reduce logging level
import matplotlib.pyplot as plt
 
#set teh styles
plt.style.use('default')

def set_doc_style():
 
    font_size=8
    matplotlib.rc('font', **{'family' : 'serif','weight' : 'normal','size'   : font_size})
     
    for k,v in {
        'axes.titlesize':font_size,
        'axes.labelsize':font_size,
        'xtick.labelsize':font_size,
        'ytick.labelsize':font_size,
        'figure.titlesize':font_size+2,
        'figure.autolayout':False,
        'figure.figsize':(17.7*cm,18*cm),#typical full-page textsize for Copernicus (with 4cm for caption)
        'legend.title_fontsize':'large',
        'text.usetex':usetex,
        }.items():
            matplotlib.rcParams[k] = v

#===============================================================================
# journal style
#===============================================================================
if env_type=='journal':
    set_doc_style() 
 
    env_kwargs=dict(
        output_format='pdf',add_stamp=False,add_subfigLabel=True,transparent=True
        )            
#===============================================================================
# draft
#===============================================================================
elif env_type=='draft':
    set_doc_style() 
 
    env_kwargs=dict(
        output_format='svg',add_stamp=True,add_subfigLabel=True,transparent=True
        )          
#===============================================================================
# presentation style    
#===============================================================================
elif env_type=='present': 
 
    env_kwargs=dict(
        output_format='svg',add_stamp=True,add_subfigLabel=False,transparent=False
        )   
 
    font_size=12
 
    matplotlib.rc('font', **{'family' : 'sans-serif','sans-serif':'Tahoma','weight' : 'normal','size':font_size})
     
     
    for k,v in {
        'axes.titlesize':font_size+2,
        'axes.labelsize':font_size+2,
        'xtick.labelsize':font_size,
        'ytick.labelsize':font_size,
        'figure.titlesize':font_size+4,
        'figure.autolayout':False,
        'figure.figsize':(34*cm,19*cm), #GFZ template slide size
        'legend.title_fontsize':'large',
        'text.usetex':usetex,
        }.items():
            matplotlib.rcParams[k] = v
  
print('loaded matplotlib %s'%matplotlib.__version__)

#===============================================================================
# IMPORTS-------
#===============================================================================

import os, hashlib, logging, shutil, psutil, webbrowser
from datetime import datetime


import numpy as np
import pandas as pd

import dask


from osgeo import gdal # Import gdal before rasterio
import rioxarray
import xarray as xr

import matplotlib.pyplot as plt
import seaborn as sns

from definitions import lib_dir, wrk_dir
from superd.hyd.ahr_params import epsg_id, scenarioTags_d

from superd.hp import init_log, today_str, get_filepaths, dstr


from superd.hp import init_log, today_str, get_filepaths
from superd.hyd.coms import load_nc_to_xarray


def calc_stats_ds(nc_fp, log):
    with xr.open_dataarray(nc_fp, engine='netcdf4', mask_and_scale=True, 
        chunks={'x':-1, 'y':-1, 'tag':1, 'MannningsValue':1}) as da:
        log.info(f'loaded {da.shape} w/ \n    {da.dims}\n    {da.chunks}')
        da = da.drop(['band', 'spatial_ref'])
        #=======================================================================
        # compute metrics
        #=======================================================================
        log.info(f'computing metrics')
        d = dict()
        
        #get count
        """not working on groupby
        d['cnt'] = da.groupby('tag').count(dim=['x', 'y']).to_series()"""
        
        count_d = {k:gda.count(dim=['x', 'y']).to_series() for k, gda in da.groupby('tag')}
        d['count'] = pd.concat(count_d, names=['tag'], axis=0) 
        
               
        d['max'] = da.groupby('tag').max(dim=['x', 'y']).to_series()
        d['mean'] = da.groupby('tag').fillna(0.0).mean(dim=['x', 'y']).to_series() #.compute()
        d['null_cnt'] = da.isnull().groupby('tag').sum(dim=['x', 'y']).to_series()
        
    #===========================================================================
    # combine
    #===========================================================================
 
    return pd.concat(d, names=['stat'], axis=1)

def plot_stats_per_sim(
        nc_fp=None,
        out_dir=None,
        stats_pick_fp=None,
        ):
    """compute stats for each hyd result in teh stack"""
    
    #===========================================================================
    # defautls
    #===========================================================================
    
    if nc_fp is None:
        
        nc_fp = get_filepaths(os.path.join(lib_dir, '01_concat'), count=1)
        
    
    if out_dir is None:
        out_dir=os.path.join(wrk_dir, 'stats_per_sim', today_str)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name='stats')
    
    #===========================================================================
    # build/load stats
    #===========================================================================
    if stats_pick_fp is None:
        dxind = calc_stats_ds(nc_fp, log)
        shape_str = '-'.join([str(e) for e in dxind.shape])
        stats_pick_fp = os.path.join(out_dir, f'stats_{shape_str}_{today_str}.pkl')
        
        dxind.to_pickle(stats_pick_fp)
        
        log.info(f'wrote {dxind.shape} to \n    {stats_pick_fp}')
        
    else:
        log.warning(f'loading dxind from \n    {stats_pick_fp}')
        dxind = pd.read_pickle(stats_pick_fp)
    
    log.info(f'finished w/ {dxind.shape})')
    shape_str = '-'.join([str(e) for e in dxind.shape])
    #===========================================================================
    # poast
    #===========================================================================
    #dxind['real_cnt'] = dxind['count']-dxind['null_cnt']
    
    #data = [dxind[k] for k in 
    res_d=dict()
    #===========================================================================
    # hexbin scatters
    #===========================================================================
 
    ax = sns.relplot(
            data=dxind.stack().rename('vals').reset_index() ,
            x="MannningsValue",
            y="vals",
            hue="tag", 
            #size="choice", 
            row="stat",
            kind="line", 
            size_order=["T1", "T2"], 
             
            height=3, aspect=3,  
            facet_kws=dict(sharey=False),
 
        )
    
    #write
    res_d['relplot'] = os.path.join(out_dir, f'stats_relplot_{len(dxind)}.svg')
    ax.figure.savefig(res_d['relplot'], dpi = 300,   transparent=True)
    log.info(f'wrote to \n    %s'%res_d['relplot'])
 
    plt.close('all')
    return
    #===========================================================================
    # violin plots
    #===========================================================================
    
    
    for varName in ['count', 'max', 'mean']:
        log.info(f'plotting {varName}')
        ax = sns.violinplot(data=dxind.reset_index('tag'), x=varName, y='tag', 
                       palette="light:g", inner="points", orient="h")
        
        ax.set_title(f'{varName} for {len(dxind)} sims')
 
        #write
        ofp = os.path.join(out_dir, f'stats_{varName}_{len(dxind)}.svg')
        ax.figure.savefig(ofp, dpi = 300,   transparent=True)
        
        plt.close('all')
        
        log.info(f'wrote to \n    {ofp}')
        res_d[f'{varName}_violin'] = ofp
        

        
    log.info(f'finished w/ \n    {dstr(res_d)}')
    
    return res_d
        
        
    
    
    
        
 
    
    
if __name__=="__main__":
    
    plot_stats_per_sim(
        stats_pick_fp=r'l:\10_IO\2307_super\stats_per_sim\20230807\stats_1495-4_20230807.pkl',
        #nc_fp=r'l:\10_IO\2307_super\lib\01_concatb\meta_raw_1494.pkl'
        )
    
    
    
    
    
    
    
    