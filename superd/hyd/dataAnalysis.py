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
        'figure.figsize':(17.7*cm,30*cm),#typical full-page textsize for Copernicus (with 4cm for caption)
        #'figure.figsize':(17.7*cm,18*cm),#typical full-page textsize for Copernicus (with 4cm for caption)
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

import os, string
from datetime import datetime


import numpy as np
import pandas as pd

from pandas import IndexSlice as idx

 


from osgeo import gdal # Import gdal before rasterio
#import rioxarray
#import xarray as xr

import matplotlib.pyplot as plt
import seaborn as sns

from definitions import wrk_dir, lib_dir
 
from superd.hyd.ahr_params import epsg_id, scenarioTags_d, coln_d

from superd.hp import init_log, today_str, get_filepaths, dstr 

 
 

from palettable.colorbrewer.diverging import BrBG_5


#===============================================================================
# HELPERS------
#===============================================================================


def get_matrix_fig(  
                       row_keys, #row labels for axis
                       col_keys, #column labels for axis (1 per column)
                       
                       fig_id=0,
                       figsize=None, #None: calc using figsize_scaler if present
                       figsize_scaler=None,
                        #tight_layout=False,
                        constrained_layout=True,
                        set_ax_title=True, #add simple axis titles to each subplot
                        log=None,
                        add_subfigLabel=True,
                        fig=None,
                        **kwargs):
        
        """get a matrix plot with consistent object access
        
        Parameters
        ---------
        figsize_scaler: int
            multipler for computing figsize from the number of col and row keys
            
        add_subfigLabel: bool
            add label to each axis (e.g., A1)
            
        Returns
        --------
        dict
            {row_key:{col_key:ax}}
            
        """
        
        
        #=======================================================================
        # defautls
        #=======================================================================
 
        #special no singluar columns
        if col_keys is None: ncols=1
        else:ncols=len(col_keys)
        
        log.info(f'building {len(row_keys)}x{len(col_keys)} fig\n    row_keys:{row_keys}\n    col_keys:{col_keys}')
        
        #=======================================================================
        # precheck
        #=======================================================================
        """needs to be lists (not dict keys)"""
        assert isinstance(row_keys, list)
        #assert isinstance(col_keys, list)
        #=======================================================================
        # build figure
        #=======================================================================
        # populate with subplots
        if fig is None:
            if figsize is None: 
                if figsize_scaler is None:
                    figsize=matplotlib.rcParams['figure.figsize']
                else:
                    
                    figsize = (len(col_keys)*figsize_scaler, len(row_keys)*figsize_scaler)
                    
                    #fancy diagnostic p rint
                    fsize_cm = tuple(('%.2f cm'%(e/cm) for e in figsize))                    
                    log.info(f'got figsize={fsize_cm} from figsize_scaler={figsize_scaler:.2f} and col_cnt={len(col_keys)}')
                    
 
                
        
            fig = plt.figure(fig_id,
                figsize=figsize,
                #tight_layout=tight_layout,
                constrained_layout=constrained_layout,
 
                )
        else:
            #check the user doesnt expect to create a new figure
            assert figsize_scaler is None
            assert figsize is None
            assert constrained_layout is None
            assert fig_id is None
        

        #=======================================================================
        # add subplots
        #=======================================================================
        ax_ar = fig.subplots(nrows=len(row_keys), ncols=ncols, **kwargs)
        
        #convert to array
        if not isinstance(ax_ar, np.ndarray):
            assert len(row_keys)==len(col_keys)
            assert len(row_keys)==1
            
            ax_ar = np.array([ax_ar])
            
        
        #=======================================================================
        # convert to dictionary 
        #=======================================================================
        ax_d = dict()
        for i, row_ar in enumerate(ax_ar.reshape(len(row_keys), len(col_keys))):
            ax_d[row_keys[i]]=dict()
            for j, ax in enumerate(row_ar.T):
                ax_d[row_keys[i]][col_keys[j]]=ax
        
                #=======================================================================
                # post format
                #=======================================================================
                if set_ax_title:
                    if col_keys[j] == '':
                        ax_title = row_keys[i]
                    else:
                        ax_title='%s.%s'%(row_keys[i], col_keys[j])
                    
                    ax.set_title(ax_title)
                    
                    
                if add_subfigLabel:
                    letter=list(string.ascii_lowercase)[j]
                    ax.text(0.05, 0.95, 
                            '(%s%s)'%(letter, i), 
                            transform=ax.transAxes, va='top', ha='left',
                            size=matplotlib.rcParams['axes.titlesize'],
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
                            )
 
                
            
 
        log.info('built %ix%i w/ figsize=%s'%(len(col_keys), len(row_keys), figsize))
        return fig, ax_d

#===============================================================================
# PLOTERS----------
#===============================================================================


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
    """compute WSH stats for each scenario as a function of Mannings"""
    
    #===========================================================================
    # defautls
    #===========================================================================
    
    if nc_fp is None:
        
        nc_fp = get_filepaths(os.path.join(lib_dir, '01_concat'), count=1)
        
    
    if out_dir is None:
        out_dir=os.path.join(wrk_dir, 'outs', 'stats_per_sim', today_str)
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
        
        
def plot_inun_perf_stack(
        df_fp = None,
        out_dir=None,
        ):
    
    """quick inun performance multi-plot with seaborn"""
    
    
    #===========================================================================
    # defaults
    #===========================================================================
    
    if out_dir is None:
        out_dir=os.path.join(wrk_dir, 'outs', 'inun_perf', today_str)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name='stats')
    res_d=dict()
    #===========================================================================
    # load
    #===========================================================================
    dxind = pd.read_pickle(df_fp)
    
    log.info(f' loaded {dxind.shape} from \n    {df_fp}')
    
    serx = dxind.stack().rename('vals')
    serx.index = serx.index.set_names('metric', level=2)
    #===========================================================================
    # plot
    #===========================================================================
    
    for i, coln_l in enumerate([
        ['criticalSuccessIndex','hitRate', 'falseAlarms',  'errorBias'],
        ['TN','TP', 'FP', 'FN']]
        ):
        
        
 
 
        facetGrid = sns.relplot(
                data=serx.loc[:, :, coln_l].reset_index(),
                x="MannningsValue",
                y="vals",
                hue="tag", 
                #size="choice", 
                row="metric",
                kind="line", 
                size_order=["T1", "T2"], 
                 
                height=3, aspect=3,  
                facet_kws=dict(sharey=False),
     
            )
    
 
        #===========================================================================
        # add optimum line
        #===========================================================================
        opt_d = dict()
        for tag, gdf in dxind.groupby('tag'):
            
            #filter those with low false alarms
            #bx = gdf['errorBias']<0.1
            bx = gdf.index.get_level_values(coln_d['man'])>0.05
 
            
            #get location of maximum CSI within this range
            opt_d[tag] = gdf.loc[bx, 'criticalSuccessIndex'].idxmax()[1]
            
            for ax_ar in facetGrid.axes:
                ax=ax_ar[0]
                ax.axvline(opt_d[tag], label=tag, color='black', linestyle='dashed', linewidth=0.5)
 
        log.info(f'got optimums: \n    {opt_d}')
        
        anno_obj = facetGrid.axes[0][0].text(0.1, 0.9, dstr(opt_d), transform=ax.transAxes, va='center',
                           fontsize=16)
        
        #===========================================================================
        # dxind['criticalSuccessIndex'].groupby('tag').idxmax()
        # 
        # dxind['criticalSuccessIndex'].idxmax()
        #===========================================================================
        
        #ax.axvline()
        
        #===========================================================================
        # #write
        #===========================================================================
        res_d['relplot'] = os.path.join(out_dir, f'stats_relplot_{i}_{len(dxind)}.svg')
        ax.figure.savefig(res_d['relplot'], dpi = 300,   transparent=True)
        log.info(f'wrote to \n    %s'%res_d['relplot'])
     
        plt.close('all')
        
        log.info('finished')
    
    """custom plot?  mark optimum?"""
        
    
    
    
def plot_inun_perf_stack2(
        df_fp = None,
        out_dir=None,
        ):
    
    """plot (2x) inundation performance metrics for each scenario as a function of Mannings
    
    Params
    --------
    df_fp: str
        filepath to res_dx pickle
        ouput of superd.hyd._05eval.compute_inundation_performance()
    
    """
    
    
    #===========================================================================
    # defaults
    #===========================================================================
    
    if out_dir is None:
        out_dir=os.path.join(wrk_dir, 'outs', 'inun_perf', today_str)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name='stats')
 
    #===========================================================================
    # load
    #===========================================================================
    dxind = pd.read_pickle(df_fp)
    
    log.info(f' loaded {dxind.shape} from \n    {df_fp}')
    

    
    #reorder levels
    dxind = dxind.loc[idx[list(scenarioTags_d.keys()), :], :]
    
    #filter bogus values
    bx = dxind.index.get_level_values(1)>0.05
    
    #===========================================================================
    # two plots
    #===========================================================================
    res_d=dict()
    for i, coln_l in enumerate([
        ['criticalSuccessIndex','hitRate', 'falseAlarms',  'errorBias'],
        ['TN','TP', 'FP', 'FN']]
        ):
        
        #build figure
        fig = get_inun_perf_fig(dxind.loc[bx, coln_l],   log=log)
        
        #write
        
        res_d[i] = os.path.join(out_dir, f'inun_perf2_{i}_{len(dxind)}.svg')
        fig.savefig(res_d[i], dpi = 300,   transparent=True)
        log.info(f'wrote to \n    %s'%res_d[i])
        
    #===========================================================================
    # wrap
    #===========================================================================
    log.info(f'finished on {len(res_d)}\n    {dstr(res_d)}')
    
    return res_d
        
def get_inun_perf_fig(dxind,                       
                         log=None,
                         #cmap = plt.cm.get_cmap(name='Set1'),
                         cmap=BrBG_5.mpl_colormap
 
                       ):
    
    
    
    #===========================================================================
    # set up figure
    #===========================================================================
    plt.close('all')
    row_keys, col_keys, mod_keys = dxind.columns.tolist(), [' '], dxind.index.get_level_values('tag')
    fig, ax_d = get_matrix_fig(row_keys, col_keys, log=log, set_ax_title=False, sharex=True)
    
    #drop level
    ax_d = {k:v[' '] for k,v in ax_d.items()}
    
    #set up color map
    
    #get a dict of key:hex-color from the list of keys and the colormap
    ik_d = dict(zip(mod_keys, np.linspace(0, 1, len(mod_keys))))    
    hex = lambda x:matplotlib.colors.rgb2hex(x)
    color_d = {k:hex(cmap(ni)) for k, ni in ik_d.items()}
    
    #over-ride base to be black
    color_d['base'] = 'black'
    
    #===========================================================================
    # identify optimum
    #===========================================================================
    if 'criticalSuccessIndex' in dxind.columns:
        plot_optimum=True
    else:
        plot_optimum=False
    

    
    #===========================================================================
    # loop and plot each
    #===========================================================================
    opt_d = dict()
    for tag, gdx in dxind.groupby('tag', group_keys=False):
        
        log.info(f'on {tag} w/ {len(gdx)}')
        gdf = gdx.droplevel(0)
        
        #=======================================================================
        # identify optimum
        #=======================================================================
        if plot_optimum:
            #filter out bogus mannings
            print(gdf.columns)
            bx = gdf.index.get_level_values(coln_d['man'])>0.05    
            
            #get location of maximum CSI within this range
            opt_d[tag] = gdf.loc[bx, 'criticalSuccessIndex'].idxmax() 
        
        #=======================================================================
        # plot
        #=======================================================================
        
        for metric, ax in ax_d.items():
            
            #plot tag.metric line 
            xar, yar = gdf.index.values, gdf[metric].values        
    
            ax.plot(xar, yar, color=color_d[tag])
            
            #plot optimum
            if plot_optimum:
                ax.axvline(opt_d[tag], label=f'{tag}={opt_d[tag]:.3f}', 
                           color=color_d[tag], linestyle='dashed', linewidth=1.0)
                
    log.info(f'finished')
    
    
    #===========================================================================
    # post
    #===========================================================================
    for row_key, ax in ax_d.items():
        
        ax.set_ylabel(row_key)
        
        #first row
        if row_key==row_keys[0]:
            ax.legend()
            
        #last row
        if row_key==row_keys[-1]:
            ax.set_xlabel('Mannings n')
    
    #===========================================================================
    # wrap
    #===========================================================================
    log.info('finished')
    
    return fig
    
 
 
    
    
if __name__=="__main__":
    
    plot_stats_per_sim(
        stats_pick_fp=r'l:\10_IO\2307_super\outs\stats_per_sim\20230807\stats_1495-4_20230807.pkl',
        #nc_fp=r'l:\10_IO\2307_super\lib\01_concatb\meta_raw_1494.pkl'
        )
    
    
    #===========================================================================
    # plot_inun_perf_stack2(
    #     df_fp=r'l:\10_IO\2307_super\outs\performance\inundation\20230823\eval_inun_metrics_1494-8_20230823.pkl',
    #     )
    #===========================================================================
    
    
    
    
    
    
    
    