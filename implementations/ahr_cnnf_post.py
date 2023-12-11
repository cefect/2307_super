'''
Created on Dec. 11, 2023

@author: cef
'''
import os
from superd.cnnf.post_performance import gtif_to_xarray, build_confusion_xr, compute_inundation_performance

from hp.basic import today_str, dstr
from hp.hyd import get_wsh_rlay
from hp.logr import get_log_stream

from definitions import wrk_dir

wrk_dir = os.path.join(wrk_dir, 'ahr')

aoi_fp=r'l:\10_IO\2307_super\ins\2207_dscale\aoi13_r32_small_0428.geojson'
dem_fp = r'l:\10_IO\2307_super\ins\2207_dscale\fdsc\dem1_clip.tif'

def _01_convert_to_wsh():
    
    #===========================================================================
    # params
    #===========================================================================
    fp_d =         {
            #'hyd_coarse':'l:\10_IO\2307_super\ins\2207_dscale\hyd_ahr_aoi13\wd_max._r32_b10_i65_0511.tif'
            
            #has some negative depths... does this make sense?
            #'rsmp':r'l:\10_IO\2307_super\ins\2207_dscale\fdsc\ahr_aoi13_0506_r2_1002_rsmp.tif',
            
            #resample w/ filter
            'rsmpF':r'l:\10_IO\2307_super\ins\2207_dscale\fdsc\ahr_aoi13_0506_r2_1002_rsmpF.tif',
            'cgs':r'l:\10_IO\2307_super\ins\2207_dscale\fdsc\ahr_aoi13_0506_r2_1002_cgs.tif',
            
            }
    
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
    #===========================================================================
    # params
    #===========================================================================
    #WSH
    fp_d =         {
            #'hyd_coarse':'l:\10_IO\2307_super\ins\2207_dscale\hyd_ahr_aoi13\wd_max._r32_b10_i65_0511.tif'
            'hyd_fine':r'l:\10_IO\2307_super\ins\2207_dscale\hyd_ahr_aoi13\wd_max_r04_b4_i05_0508.tif',
            'cnnf':r'l:\10_IO\2307_super\deploy\1209_v04b\1209_v04b_pred_wsh_patch160_floor0.10_20231211_d536dffd6be7f0dd.tif',
            'rsmpF':r'l:\10_IO\2307_super\ahr\_01_convert_to_wsh\rsmpF_WSH_20231211.tif',
            'cgs':r'l:\10_IO\2307_super\ahr\_01_convert_to_wsh\cgs_WSH_20231211.tif',
            
            }
    
    #===========================================================================
    # convert WSE to WSH
    #===========================================================================
    
    
    
    out_dir = os.path.join(wrk_dir, '01_convert_to_xr')
    
    return gtif_to_xarray(fp_d,
                          aoi_fp=aoi_fp,
                          out_dir=out_dir)
    

def _03_build_confusion(
        nc_fp = r'l:\10_IO\2307_super\ahr\01_convert_to_xr\wsh_concat_xr_20231211.nc'
        ):
    
    
    out_dir = os.path.join(wrk_dir, '03_build_confusion')
    
    return build_confusion_xr(nc_fp, out_dir=out_dir)
    
    
def _04_inundation_performance(
        nc_fp = r'l:\10_IO\2307_super\ahr\03_build_confusion\confusion_xr_20231211.nc',
        ):
    
    out_dir = os.path.join(wrk_dir, '04_inundation_performance')
    
    return compute_inundation_performance(nc_fp, out_dir=out_dir)
    
    

if __name__=="__main__":
    
    #_01_convert_to_wsh()
    
    #_02_convert_to_xr()
    
    #_03_build_confusion()
    
    _04_inundation_performance()
    
    
    
    
    