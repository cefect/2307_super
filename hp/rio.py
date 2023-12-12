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

from parameters import confusion_codes

cm = 1 / 2.54

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

def get_bbox(rlay_obj):
    bounds = get_ds_attr(rlay_obj, 'bounds')
    return sgeo.box(*bounds)


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
    # setup
    #===========================================================================
    def get_matrix_fig(self, #conveneince for getting 
                       row_keys, #row labels for axis
                       col_keys, #column labels for axis (1 per column)
                       
                       fig_id=0,
                       figsize=None, #None: calc using figsize_scaler if present
                       figsize_scaler=None,
                        #tight_layout=False,
                        constrained_layout=True,
                        set_ax_title=True, #add simple axis titles to each subplot
                        logger=None,
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
        if logger is None: logger=self.logger
        log=logger.getChild('get_mat_fig')
 
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
    def _ax_imshow(self, ax,  ar_raw, 
                        bbox=None,
                        gridk=None,
                        show_kwargs=None,
                         **kwargs):
        """add a styleized raster to the axis
        
        show_kwargs: dict
            kwargs to pass to rio.plot.show
            if None: loaded from grid_styles_lib (w/ gridk) 
        """
 
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
 
        return ax.imshow(ar, interpolation='nearest',**show_kwargs, **kwargs)
        
        
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
        
    
#===============================================================================
# ASSERTIONS-------
#===============================================================================
def assert_spatial_equal(left, right,  msg='',): 
    """check all spatial attributes match"""
    if not __debug__: # true if Python was not started with an -O option
        return 
    __tracebackhide__ = True     
    
 
    f= lambda ds, att_l=['crs', 'height', 'width', 'bounds', 'res']:_get_meta(ds, att_l=att_l)
    
    ld = rlay_apply(left, f)
    rd = rlay_apply(right, f)
 
    for k, lval in ld.items():
        rval = rd[k]
        if not lval == rval:
            raise AssertionError(f'{k} mismatch\n    right={rval}\n    left={lval}\n' + msg)