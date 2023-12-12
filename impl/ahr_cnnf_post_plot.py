'''
Created on Dec. 10, 2023

@author: cef

Ahr case study CNNF  post plots
'''


#===============================================================================
# PLOT ENV------
#===============================================================================
 
env_type = 'poster'
cm = 1 / 2.54 


           
#===============================================================================
# draft-------
#===============================================================================
if env_type=='draft':
    output_format='png'
    font_size=6
    params_d = {
        'savefig.dpi':300,
        'text.usetex':False,
        'savefig.edgecolor':'blue',
        'savefig.transparent':False        
        }

#===============================================================================
# journal style------
#===============================================================================
elif env_type=='journal':
    output_format='png'
    font_size=6
    params_d = {
        'savefig.dpi':600,
        'text.usetex':True,        
        }
       
#===============================================================================
# presentation style---------
#===============================================================================
elif env_type=='present': 
 
    font_size=14
    params_d = {
        'figure.figsize':(22*cm,18*cm), #GFZ template slide size (w,h)
        'text.usetex':False,
        }
    
elif env_type=='poster':
    output_format='png'
    font_size=8
    params_d = {
        'savefig.dpi':600,
        'text.usetex':False,        
        }
    
            
#===============================================================================
# apply
#===============================================================================
#ammend customs to defaults
params_d = {**{
        'axes.titlesize':font_size,
        'axes.labelsize':font_size,
        'xtick.labelsize':font_size,
        'ytick.labelsize':font_size,
        'figure.titlesize':font_size+2,
        'figure.autolayout':False,
        'figure.figsize':(18*cm,18*cm),#typical full-page textsize for Copernicus and wiley (with 4cm for caption)
        'legend.title_fontsize':'large',
        'text.usetex':False,
        'savefig.dpi':300,
        'legend.fontsize':font_size,
        'savefig.transparent':True,
        }, **params_d}

import matplotlib
#matplotlib.use('Qt5Agg') #sets the backend (case sensitive)
matplotlib.set_loglevel("info") #reduce logging level
import matplotlib.pyplot as plt
 
#set teh styles
plt.style.use('default')



matplotlib.rc('font', **{'family' : 'serif','weight' : 'normal','size': font_size})
matplotlib.rcParams.update(params_d)
  
print('loaded matplotlib %s'%matplotlib.__version__)


#===============================================================================
# imports-----------
#===============================================================================
import os, webbrowser
import pandas as pd
import xarray as xr
idx = pd.IndexSlice

from superd.cnnf.post_plots import Plot_inun_peformance, Plot_grids
from impl.ahr_cnnf_post import wse_fp_d, inun_fp, dem_fp
from definitions import wrk_dir
wrk_dir = os.path.join(wrk_dir, 'ahr')

aoi_box_fp=r'l:\10_IO\2307_super\ins\2207_dscale\aoi09t_zoom0308_4647.geojson'

rowLabels_d = {
    'cgs':'CostGrow', 'cnnf':'CNNFlood', 'hyd_fine':'Hydro. (s1)','rsmpF':'Resample',
    'hyd_coarse':'Hydro. (s2)'
    }

def plot_inun_perf(
        metric_fp = r'l:\10_IO\2307_super\ahr\06_concat\performance_dxcol_04_20231211.pkl',
        xr_fp= r'l:\10_IO\2307_super\ahr\03_build_confusion\grids_xr_20231211.nc',
 
        ):
    """inundation performance plot"""
    
    out_dir = os.path.join(wrk_dir, 'outs', 'post_plot')
    #===========================================================================
    # load  data
    #===========================================================================
    
    metric_df = pd.read_pickle(metric_fp).xs('inun', level=0, axis=1)
    
    with xr.open_dataarray(xr_fp, engine='netcdf4') as ds:
 
    
    
        ofp = Plot_inun_peformance().plot(metric_df, ds, 
                                          output_format=output_format,
                                     out_dir=out_dir, rowLabels_d=rowLabels_d,
                                     box_fp=aoi_box_fp,
                                     )
        
    webbrowser.open(ofp)
    
    plt.close('all')
    

    
def plot_grids(
        aoi_fp=aoi_box_fp
        ):
    
    fp_d = {k:wse_fp_d[k] for k in ['hyd_coarse', 'rsmpF','cnnf', 'cgs', 'hyd_fine']}
    
    
    
    ofp = Plot_grids().plot(fp_d,
        aoi_fp=aoi_fp, inun_fp=inun_fp, dem_fp=dem_fp,
        gridk='WSE', rowLabels_d=rowLabels_d, 
        out_dir = os.path.join(wrk_dir, 'outs', 'post_plot'),
        output_format=output_format,
        )
    
    plt.close('all')
    webbrowser.open(ofp)

if __name__=="__main__":
    plot_inun_perf()
    
    plot_grids()
    
    
    
    
    
    
    
    