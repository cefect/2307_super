'''
Created on Dec. 11, 2023

@author: cef
'''

import os
import rasterio as rio
import rasterio.plot
import numpy as np
import numpy.ma as ma

from hp.logr import get_log_stream
from hp.rio import assert_spatial_equal
from definitions import tmp_dir

def get_wsh_rlay(wse_fp, dem_fp,  out_dir = None, ofp=None, log=None,):
    """add dem and wse to get a depth grid (dry filtered)"""
    
    if log is None: log  = get_log_stream('gtif_to_xarray') #get the root logger
    
    log.info(f'buidling WSH raster from {os.path.basename(wse_fp)}')
    
    assert_spatial_equal(wse_fp, dem_fp)
    # Load DEM w/ rasterio
    with rio.open(dem_fp, 'r') as dem:
        dem_mar = dem.read(1, masked=True)
        dem_profile = dem.profile
        #dem_mask = dem.read_masks(1)

    # Load wse w/ rasterio
    with rio.open(wse_fp, 'r') as wse:
        wse_mar = wse.read(1, masked=True)
        wse_profile = wse.profile
        #wse_mask = wse.read_masks(1)
        
    assert np.any(wse_mar.mask)

    # Check the grids are the same shape and have the same profile
    #assert dem_mar.shape == wse_arr.shape, "DEM and WSE grids must have the same shape"
    assert dem_profile == wse_profile, "DEM and WSE must have the same profile"

    # Add the arrays together 
    wsh_ar = np.where(wse_mar.mask, 0, wse_mar.data - dem_mar.data)
    
    if np.any(wsh_ar<0.0):
        log.warning(f'got negative depths')
    
    wsh_mar = ma.array(wsh_ar, mask=dem_mar.mask, fill_value=dem_profile['nodata'])
    
    
    """
    import matplotlib.pyplot as plt
    from matplotlib.image import AxesImage
    plt.close('all')
    
    ax = rasterio.plot.show(wse_mar)
    img = [obj for obj in ax.get_children() if isinstance(obj, AxesImage)][0]
    
    plt.colorbar(img)
    """
    
 

    # Write the result to GeoTiff
    
    if ofp is None:
        if out_dir is not None:
            out_dir = tmp_dir
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        ofp = out_dir + "/depth_grid.tif"
        
    with rio.open(ofp, 'w', **dem_profile) as dst:
        dst.write(wsh_mar, 1, masked=False)
        
    log.info(f'finished on \n    {ofp}')
    
    return ofp

    
    