'''
Created on Dec. 11, 2023

@author: cef
'''
import os
from superd.cnnf.post_performance import (
    gtif_to_xarray, build_confusion_xr, inundation_performance,
    hwm_performance
    )

import pandas as pd

from hp.basic import today_str, dstr
from hp.hyd import get_wsh_rlay
from hp.logr import get_log_stream

from definitions import wrk_dir

wrk_dir = os.path.join(wrk_dir, 'ahr')

aoi_fp=r'l:\10_IO\2307_super\ins\2207_dscale\aoi13_r32_small_0428.geojson'
dem_fp = r'l:\10_IO\2307_super\ins\2207_dscale\fdsc\dem1_clip.tif'
inun_fp = r'l:\10_IO\2307_super\ins\2207_dscale\obsv\RLP_LfU_HQ_extrm_WCS_20230324_ahr_4647_aoi13.geojson'

#WSH
wsh_fp_d = {
            #'hyd_coarse':'l:\10_IO\2307_super\ins\2207_dscale\hyd_ahr_aoi13\wd_max._r32_b10_i65_0511.tif'
            'hyd_fine':r'l:\10_IO\2307_super\ins\2207_dscale\hyd_ahr_aoi13\wd_max_r04_b4_i05_0508.tif',
            'cnnf':r'l:\10_IO\2307_super\deploy\1209_v04b\1209_v04b_pred_wsh_patch160_floor0.10_20231211_d536dffd6be7f0dd.tif',
            'rsmpF': 'l:\\10_IO\\2307_super\\ahr\\01_convert_to_wsh\\rsmpF_WSH_20231211.tif', 
            'cgs': 'l:\\10_IO\\2307_super\\ahr\\01_convert_to_wsh\\cgs_WSH_20231211.tif'}

wse_fp_d = {
    'hyd_fine':r'l:\10_IO\2307_super\ins\2207_dscale\hyd_ahr_aoi13\wd_max_WSE_clip_r04_b4_i05_0508.tif',
    'hyd_coarse':r'l:\10_IO\2307_super\ins\2207_dscale\hyd_ahr_aoi13\wd_max_WSE_r32_b10_i65_0511.tif',
    'cnnf':r'l:\10_IO\2307_super\deploy\1209_v04b\1209_v04b_pred_wsh_patch160_floor0.10_20231211_d536dffd6be7f0dd_WSE.tif',
    'rsmp':r'l:\10_IO\2307_super\ins\2207_dscale\fdsc\ahr_aoi13_0506_r2_1002_rsmp.tif',
    'rsmpF':r'l:\10_IO\2307_super\ins\2207_dscale\fdsc\ahr_aoi13_0506_r2_1002_rsmpF.tif',
    'cgs':r'l:\10_IO\2307_super\ins\2207_dscale\fdsc\ahr_aoi13_0506_r2_1002_cgs.tif',
    }

def _01_convert_to_wsh():
    
    #===========================================================================
    # params
    #===========================================================================
    fp_d = {k:wse_fp_d[k] for k in ['cgs', 'rsmpF']}
    #===========================================================================
    # setup
    #===========================================================================
    log  = get_log_stream('_01_convert_to_wsh') #get the root logger
    d = dict()
    out_dir = os.path.join(wrk_dir, '01_convert_to_wsh')
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    #===========================================================================
    # loop on each
    #===========================================================================
    for scen, wse_fp in fp_d.items():
        d[scen] = get_wsh_rlay(wse_fp, dem_fp, log=log, out_dir=out_dir,
                               ofp=os.path.join(out_dir,  f'{scen}_WSH_{today_str}.tif'))
        
    log.info(d)
    
    return d
        
    
    


def _02_convert_to_xr():    
    out_dir = os.path.join(wrk_dir, '02_convert_to_xr')
    
    return gtif_to_xarray(wsh_fp_d,
                          aoi_fp=aoi_fp,
                          out_dir=out_dir)
    

def _03_build_confusion(nc_fp):
    
    out_dir = os.path.join(wrk_dir, '03_build_confusion')
    
    return build_confusion_xr(nc_fp, out_dir=out_dir)
    
    
def _04_inundation_performance(nc_fp):
    
    out_dir = os.path.join(wrk_dir, '04_inundation_performance')
    
    return inundation_performance(nc_fp, out_dir=out_dir)
    
def _05_hwm_performance(
         hwm_fp = r'l:\10_IO\2307_super\ins\2207_dscale\obsv\NWR_ahr11_hwm_20220113b_fix_aoi13.geojson'
        ):
    
    out_dir = os.path.join(wrk_dir, '05_hwm_performance')
    
    log  = get_log_stream('hwm_performance')
    
    err_lib, hwm_fp_d=dict(), dict()
    for tag, fp in wsh_fp_d.items():
        err_lib[tag], hwm_fp_d[tag]= hwm_performance(fp, hwm_fp, log=log)
    
    df = pd.DataFrame.from_dict(err_lib).T 
    
    #===========================================================================
    # wrap
    #===========================================================================
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    ofp = os.path.join(out_dir, f'hwm_performance_{len(df):02d}_{today_str}.pkl')
    df.to_pickle(ofp)
    
    log.info(f'finished and wrote {df.shape} to \n    {ofp}')
    return ofp

 

def _06_concat(
        inun_fp=r'l:\10_IO\2307_super\ahr\04_inundation_performance\inundation_performance_20231211.pkl',
        hwm_fp=r'l:\10_IO\2307_super\ahr\05_hwm_performance\hwm_performance_04_20231211.pkl'
        ):
    """join metrics togehter"""
    
    out_dir = os.path.join(wrk_dir, '06_concat')
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    
    
    log  = get_log_stream('06_concat')
    
    #===========================================================================
    # load
    #===========================================================================
    
    inun_df = pd.read_pickle(inun_fp)
    hwm_df = pd.read_pickle(hwm_fp)
    
    dxcol = pd.concat({'inun':inun_df, 'hwm':hwm_df}, names=['perf', 'tag'], axis=1)
    
    #===========================================================================
    # write
    #===========================================================================
    ofp = os.path.join(out_dir, f'performance_dxcol_{len(dxcol):02d}_{today_str}.pkl')
    dxcol.to_pickle(ofp)
    
    log.info(f'finished w/ {dxcol.shape}\n    {ofp}')
    
    return ofp
    

if __name__=="__main__":
    
    #_01_convert_to_wsh()
    
    #_02_convert_to_xr()
    
    #_03_build_confusion(r'l:\10_IO\2307_super\ahr\02_convert_to_xr\wsh_concat_xr_20231211.nc')
    
    #_04_inundation_performance(r'l:\10_IO\2307_super\ahr\03_build_confusion\grids_xr_20231211.nc')
    
    #_05_hwm_performance()
    
    _06_concat(
    inun_fp=r'l:\10_IO\2307_super\ahr\04_inundation_performance\inundation_performance_20231211.pkl'
    )
    
    
    
    