'''
Created on Mar. 20, 2023

@author: cefect


raster plotting
    see also: https://github.com/cefect/coms/blob/main/hp/rio_plot.py

requires these special packages
    matplotlib_scalebar: scale bar
    palettable: color scales
    earthpy: hillshade
'''

import os, string
import numpy as np

from osgeo import gdal # Import gdal before rasterio
import rasterio as rio
from rasterio.plot import show


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from matplotlib.patches import ArrowStyle

from matplotlib_scalebar.scalebar import ScaleBar
import earthpy.spatial as es #for plotting hillshades

#colormaps
"""
https://jiffyclub.github.io/palettable/colorbrewer/sequential/#
"""
from palettable.colorbrewer.sequential import BuPu_3, Blues_9, Purples_9
from palettable.colorbrewer.diverging import RdBu_3



#===============================================================================
# RASTERIO---------
#===============================================================================
def is_raster_file(filepath):
    """probably some more sophisticated way to do this... but I always use tifs"""
    assert isinstance(filepath, str)
    _, ext = os.path.splitext(filepath)
    return ext in ['.tif', '.asc']

def rlay_apply(rlay, func, **kwargs):
    """flexible apply a function to either a filepath or a rio ds"""
    
    assert not rlay is None 
    assert is_raster_file(rlay)
     
    with rio.open(rlay, mode='r') as ds:
        return func(ds, **kwargs)
 

def get_meta(rlay, **kwargs):
    return rlay_apply(rlay, lambda x:_get_meta(x, **kwargs))


def _get_meta(ds, att_l=['crs', 'height', 'width', 'transform', 'nodata', 'bounds', 'res', 'dtypes']):
    d = dict()
    for attn in att_l:        
        d[attn] = getattr(ds, attn)
    return d


class RioPlotr(object):
    """grid plotting
    
    TODO: combine this with coms.hyd.HydTypes"""
    
    
    #standard styles for grids
    """
    WSH and WSE colors:
        best to avoid complex color scales that might interfere with other colors  you're using
            scales w/ more sequences (e.g., Blues_9 vs Blues_3) provide greater contrast
        some scenarios maybe it's nice to use the fancier scales?
    """
    grid_styles_lib={
        'hillshade':dict(cmap=plt.cm.copper, norm=None),
        'DEM':dict(cmap = 'plasma', norm = None),
        'WSH':dict(cmap = Blues_9.mpl_colormap, norm = matplotlib.colors.Normalize(vmin=0, vmax=4)),
        'WSE':dict(cmap = Purples_9.mpl_colormap, norm = None), #May lead to inconsistent color styles
        'DIFF':dict(cmap = RdBu_3.mpl_colormap, norm = matplotlib.colors.Normalize(vmin=-0.5, vmax=0.5)),
        }
    
    #confusion colors
    #https://colorbrewer2.org/?type=diverging&scheme=RdYlBu&n=4#type=diverging&scheme=PiYG&n=4
    """spent some time thinking about this
    want to avoid blues... because almost everything is inundation. just show a WSH grid in parallel.
    colorblind safe
    lots of TrueNegatives... generally masking these out
    """
    confusion_color_d = {
            'FN':'#f1b6da', 'FP':'#d01c8b', 'TP':'#b8e186', 'TN':'white'
            }
    
    def __init__(self, **kwargs):
        
        #super().__init__(**kwargs)
        self.confusion_codes=confusion_codes
        self._set_confusion_style()
        
    
    #===========================================================================
    # style---------
    #===========================================================================
    def _set_confusion_style(self):
        """setup styles for confusion plotting"""
        
        confusion_color_d=self.confusion_color_d.copy()
        cc_d = self.confusion_codes.copy()
        
        #get rastetr val to color conversion for confusion grid
        cval_d = {v:confusion_color_d[k] for k,v in cc_d.items()}        
        cval_d = {k:cval_d[k] for k in sorted(cval_d)} #sort it
        self.confusion_val_d = cval_d.copy()
        
        cmap = matplotlib.colors.ListedColormap(cval_d.values())        
        norm = matplotlib.colors.BoundaryNorm(
                                    np.array([0]+list(cval_d.keys()))+1, #bounds tt capture the data 
                                      ncolors=len(cval_d),
                                      #cmap.N, 
                                      extend='neither',
                                      clip=True,
                                      )
        
        self.grid_styles_lib['CONFU'] = {'cmap':cmap, 'norm':norm}
        
    
    def _mask_grid_by_key(self, ar_raw, gridk, cc_d=None):
        """apply a mask to the grid based on the grid type"""
        if cc_d is None: cc_d=self.confusion_codes.copy()
        
        if gridk=='WSH':
            assert np.any(ar_raw == 0), 'depth grid has no zeros '
            ar = np.where(ar_raw == 0, np.nan, ar_raw)
        elif 'CONFU' in gridk:
            # mask out true negatives
            ar = np.where(ar_raw == cc_d['TN'], np.nan, ar_raw)
        elif 'DEM' == gridk:
            ar = np.where(ar_raw < 130, ar_raw, np.nan)
 
        elif 'WSE' in gridk: #no masking
            ar = ar_raw
        elif 'hillshade'==gridk:
            ar = es.hillshade(ar_raw)
        elif 'DIFF' in gridk:
            assert np.any(ar_raw == 0), 'DIFF grid has no zeros '
            ar = np.where(ar_raw == 0, np.nan, ar_raw)
        else:
            raise KeyError(gridk)
        return ar
    
    def _get_colorbar_pars_by_key(self, gridk):
        """get standard colorbar parameters based on the grid type"""
        if gridk=='WSH':
            spacing = 'proportional'
            label = 'WSH (m)'
            fmt = matplotlib.ticker.FuncFormatter(lambda x, p:'%.1f' % x)
            location = 'bottom'
        elif 'CONFU' in gridk:
            #spacing='proportional'
            spacing = 'uniform'
            label = 'Confusion'
            fmt = None
            #fmt = matplotlib.ticker.FuncFormatter(lambda x, p:cc_di[x])
            #cax=cax_bot
            location = 'bottom'
        elif 'DEM' in gridk:
            spacing = 'proportional'
            label = 'DEM (masl)'
            fmt = matplotlib.ticker.FuncFormatter(lambda x, p:'%.0f' % x)
            location = 'bottom'
 
            
        elif 'WSE' in gridk:
            spacing = 'proportional'
            fmt = matplotlib.ticker.FuncFormatter(lambda x, p:'%.1f' % x)
            label = 'WSE (masl)'
            location='bottom'
            
        elif 'DIFF' in gridk:
            spacing = 'proportional'
            fmt = matplotlib.ticker.FuncFormatter(lambda x, p:'%.1f' % x)
            label = 'difference (m)'
            location='bottom'
            
 
        else:
            raise KeyError( gridk)
            
        return location, fmt, label, spacing
    
    #===========================================================================
    # Widgets--------
    #===========================================================================-
    def _add_scaleBar_northArrow(self, ax,
                                 xy_loc=(0.2, 0.01),
 
                                 ):
        """add a scalebar and a north arrow to an axis
        
        
        plt.show()
        """
        scaleBar_artist = ax.add_artist(ScaleBar(1.0, "m", length_fraction=0.2, location='lower left', 
                box_alpha=0.8,border_pad=0.3
                #frameon=False,
                ))
        
 
 
    
        #=======================================================================
        # #north arrow
        #=======================================================================
        """spent 30mins on this... couldnt find an elegent way"""
  #=============================================================================
  #       xytext=(xy_loc[0], xy_loc[1]+0.15)
  #       ax.annotate('', xy=xy_loc, xytext=xytext,
  #           arrowprops=dict(facecolor='black', 
  #                           linewidth=1.5,  
  #                           #headwidth=8,
  #                           arrowstyle='<-',
  # 
  #                           #arrowstyle=ArrowStyle("]-", head_length=.4, head_width=.4, tail_width=.4),
  #                           ),
  #           ha='center', va='center', #fontsize=16,
  #           #xycoords=ax.transAxes,
  #           xycoords='axes fraction',textcoords='axes fraction',
  #           )
  #        
  #       #add the N
  #       xytext2 = (xytext[0], xytext[1]+0.03)
  #       ax.annotate('N', xy=xytext2,ha='center', va='top', #fontsize=16,
  #           #xycoords=ax.transAxes,
  #           xycoords='axes fraction',textcoords='axes fraction',
  #           )
  #=============================================================================
        
 
        
    #===========================================================================
    # plotters------
    #===========================================================================
    def _ax_raster_show(self, ax,  fp, 
                        bbox=None,
                        gridk=None,
                        show_kwargs=None,
                         **kwargs):
        """add a styleized raster to the axis
        
        show_kwargs: dict
            kwargs to pass to rio.plot.show
            if None: loaded from grid_styles_lib (w/ gridk) 
        """
        #=======================================================================
        # defaults
        #=======================================================================
 
        with rio.open(fp, mode='r') as ds:
            
            #===================================================================
            # #load and clip the array
            #===================================================================
            if bbox is None:
                window = None
                transform = ds.transform
            else:
                window = rio.windows.from_bounds(*bbox.bounds, transform=ds.transform)
                #transform = rio.transform.from_bounds(*bbox.bounds, *window.shape)
                transform = rio.windows.transform(window, ds.transform)
                
            ar_raw = ds.read(1, window=window, masked=True)
            #===========================================================
            # #apply masks
            #===========================================================
            ar = self._mask_grid_by_key(ar_raw, gridk)
            #===========================================================
            # #get styles by key
            #===========================================================
            if show_kwargs is None:
                if gridk is None:
                    show_kwargs=dict() 
                else:
                    assert gridk in self.grid_styles_lib, f'no gridk \'{gridk}\' found in style lib'
                    show_kwargs = self.grid_styles_lib[gridk]
                    
            assert isinstance(show_kwargs, dict)
        #===========================================================
        # plot it
        #===========================================================
        """
        plt.show()
        """
 
        return show(ar, 
                    transform=transform, 
                    ax=ax, contour=False,interpolation='nearest',**show_kwargs, **kwargs)