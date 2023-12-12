"""main caller for CNNF post plots"""

 




#===============================================================================
# imports---------
#===============================================================================
import os, copy
from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage

import fiona
import shapely.geometry as sgeo
from pyproj.crs import CRS
import geopandas as gpd


from hp.basic import get_dict_str, today_str
from hp.logr import get_log_stream
from hp.rio import RioPlotr, get_bbox, get_meta
from definitions import wrk_dir



def get_bbox_and_crs(fp):
    with fiona.open(fp, "r") as source:
        bbox = sgeo.box(*source.bounds) 
        crs = CRS(source.crs['init'])
        
    return bbox, crs


class Plot_grids(RioPlotr):
    """worker for plotting raw raster  results"""
 
    def plot(self,
                          fp_d,
                          gridk='WSE', #for formatting by grid type
                          mod_keys=None,
                          
                          dem_fp=None,
                          inun_fp=None,
                          aoi_fp=None,
                          
                          output_format='svg',rowLabels_d=None,
                          colorBar_bottom_height=0.15,
 
                          vmin=None, vmax=None,
                          show_kwargs=None,
                          fig_mat_kwargs=dict(add_subfigLabel=True),
                          arrow_kwargs_lib=dict(),
                          inun_kwargs = dict(facecolor='none', 
                                             edgecolor='black', 
                                             linewidth=0.75, 
                                             linestyle='dashed'),
                          log=None,out_dir=None, ofp=None,):
        """matrix plot of raster results. nice for showing a small region of multiple sims
        
        Pars
        --------
        fp_lib: dict
            filepaths of grids for plotting
                {modk:{gridk ('dem', 'true_inun', ...):fp}}
        gridk: str, default: 'pred_wse'
            which grid to plot
            
        add_subfigLabel: bool
            True: add journal sub-figure labelling (a0)
            False: use the fancy labels
            
        show_kwargs: dict
            over-ride default grid show kwargs (should be a lvl2 dict?)
            
        """
        
        #===========================================================================
        # setup
        #===========================================================================
        start = datetime.now() 
        #configure outputs
        if ofp is  None:
            if out_dir is None:
                out_dir = os.path.join(wrk_dir, 'outs', 'cnnf', 'post_plots')
         
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                
            ofp = os.path.join(out_dir, f'plot_grids_{gridk}_{today_str}.{output_format}')
            
        if log is None: log  = get_log_stream('plot_inun_perf') #get the root logger
        
        #=======================================================================
        # defaults
        #=======================================================================
 
 
        
        #list of model values
        if mod_keys is None:
            mod_keys = list(fp_d.keys())
            #mod_keys = ['WSE2', 'Basic', 'SimpleFilter', 'Schumann14', 'CostGrow','WSE1']            
        assert set(mod_keys).difference(fp_d.keys())==set()
        
        #model fancy labels
        if rowLabels_d is None:
            rowLabels_d=self.rowLabels_d
            
        if rowLabels_d is None:
            rowLabels_d = dict()
            
        #add any missing
        for k in mod_keys:
            if not k in rowLabels_d:
                rowLabels_d[k] = k
            
        #bounding box
        rmeta_d = get_meta(dem_fp)  #spatial meta from dem for working with points
        
        if aoi_fp is None:
            bbox, crs=get_bbox(dem_fp), self.crs
        else:            
            bbox, crs=get_bbox_and_crs(aoi_fp)
            log.info(f'using aoi from \'{os.path.basename(aoi_fp)}\'')
            
        
        
               
        assert crs.to_epsg()==rmeta_d['crs'].to_epsg()
            
        log.info(f'plotting {len(fp_d)} on {mod_keys}')
        #=======================================================================
        # setup figure
        #=======================================================================        
        ax_d, mat_df, row_keys, col_keys, fig = self._get_fig_mat_models(                                            
                                            mod_keys,
                                            logger=log, ncols=3,
                                            constrained_layout=False, 
                                            **fig_mat_kwargs)
 
        
        #=======================================================================
        # plot loop------
        #=======================================================================        
        meta_lib, axImg_d=dict(), dict()
        for rowk, d0 in ax_d.items():
            for colk, ax in d0.items():                
                #===============================================================
                # setup
                #===============================================================
                modk = mat_df.loc[rowk, colk]
                if modk=='nan': 
                    ax.axis('off')
                    continue              
                log.info(f'plotting {rowk}x{colk} ({modk})\n    {fp_d[modk]}')
                
                #===============================================================
                # plot it
                #===============================================================
                # DEM raster 
                self._ax_raster_show(ax,  dem_fp, bbox=bbox,gridk='hillshade')
                
                # focal raster
                fp = fp_d[modk]
                self._ax_raster_show(ax,  fp, bbox=bbox,gridk=gridk, alpha=0.9, show_kwargs=show_kwargs,
                                     vmin=vmin, vmax=vmax)
                
                #log.debug(get_data_stats(fp))
                #inundation                
                gdf = gpd.read_file(inun_fp)
                assert gdf.geometry.crs==crs, f'crs mismatch: {gdf.geometry.crs}\n    {inun_fp}'
 
                #boundary 
                gdf.clip(bbox.bounds).plot(ax=ax,**inun_kwargs)
 
                #===============================================================
                # label
                #=============================================================== 
                ax.text(0.95, 0.05, 
                            rowLabels_d[modk], 
                            transform=ax.transAxes, va='bottom', ha='right',
                            size=matplotlib.rcParams['axes.titlesize'],
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
                            )
                
                #===============================================================
                # #scale bar
                #===============================================================
                self._add_scaleBar_northArrow(ax)
                    
                #===============================================================
                # wrap
                #===============================================================
 
                # hide labels
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                
        #=======================================================================
        # add annotation arrows------
        #=======================================================================
        """
        plt.show()
        """
        if len(arrow_kwargs_lib)>0:
            for k, arrow_kwargs in arrow_kwargs_lib.items():
                self._add_arrow(ax_d, logger=log, **arrow_kwargs)
        #=======================================================================
        # colorbar-------
        #=======================================================================
        #grab the image object for making the colorbar
        ax = ax_d[row_keys[0]][col_keys[0]] #use the first axis
        l= [obj for obj in ax.get_children() if isinstance(obj, AxesImage)]
        axImg_d = dict(zip(['dem', gridk], l))
                
        log.debug(f'adding colorbar')
        
        #get parameters
        _, fmt, label, spacing = self._get_colorbar_pars_by_key(gridk)
        shared_kwargs = dict(orientation='horizontal',
                             extend='both', #pointed ends
                             shrink=0.8,
                             ticklocation='top',
                             )
        
        #add the new axis
        fig.subplots_adjust(bottom=colorBar_bottom_height, 
                            wspace=0.05, top=0.999, hspace=0.05, left=0.05, right=0.95) 
        
        cax = fig.add_axes((0.1, 0.01, 0.8, 0.03)) #left, bottom, width, height
        
        #add the bar
        cbar = fig.colorbar(axImg_d[gridk],cax=cax,label=label,format=fmt, spacing=spacing,
                                 **shared_kwargs)
        
        
        
        #=======================================================================
        # post
        #=======================================================================
        """nothing in the legend for some reason...."""
        for rowk, d0 in ax_d.items():
            for colk, ax in d0.items():
                 
                if rowk==row_keys[0]:
                    if colk==col_keys[-1]:
                        dummy_patch = matplotlib.patches.Patch(label='observed', **inun_kwargs)
                        ax.legend(handles=[dummy_patch], loc='upper right')
                        

 
                        
        #=======================================================================
        # wrap
        #=======================================================================
        
        
        fig.savefig(ofp, dpi = 300, format = output_format, transparent=True)
        
        log.info(f'finished and wrote figure to \n    {ofp}')
                    
        return ofp
    
    def _get_fig_mat_models(self, mod_keys, 
                            ncols=1,  
                            total_fig_width=None, 
                            figsize=None, 
                            constrained_layout=True, 
                            **kwargs):
        #=======================================================================
        # defaults
        #=======================================================================
 
        if total_fig_width is  None: total_fig_width = matplotlib.rcParams['figure.figsize'][0]
        #=======================================================================
        # #reshape into a frame
        #=======================================================================
        #bad division
        mod_keys2 = copy.copy(mod_keys)
 
        if len(mod_keys) % ncols != 0:
            for i in range(ncols - len(mod_keys) % ncols):
                mod_keys2.append(np.nan)
            
 
        
        mat_df = pd.DataFrame(np.array(mod_keys2).reshape(-1, ncols))
        mat_df.columns = [f'c{e}' for e in mat_df.columns]
        mat_df.index = [f'r{e}' for e in mat_df.index]
        row_keys = mat_df.index.tolist()
        
        col_keys = mat_df.columns.tolist()
        
        #figure size
        if figsize is None:
            figsize_scaler = (total_fig_width / ncols)
        else:
            #assert ncols is None
            assert total_fig_width is None
            figsize_scaler = None
            
        fig, ax_d = self.get_matrix_fig(row_keys, col_keys, 
            set_ax_title=False, 
            figsize=figsize, 
            constrained_layout=constrained_layout, 
            sharex='all', 
            sharey='all', 
            #add_subfigLabel=True, 
            figsize_scaler=figsize_scaler, 
            **kwargs)
        
        return ax_d, mat_df, row_keys, col_keys, fig
        
 
        


 



class Plot_inun_peformance(RioPlotr):
    """worker for plotting inundation performance of downscaling and hydro methods
    
    see also fperf.plot.inun
    """
    gdf_d=dict() #container for preloading geopandas
    
    metric_labels = {'hitRate': 'Hit Rate', 
                     'falseAlarms':'False Alarms', 
                     'criticalSuccessIndex':'Crit. Suc. Index', 
                     'errorBias':'Error Bias',
                     }
    
 
    def plot(self,
                      metric_df,
                      grid_ds, 
 
                      output_format='svg',
                      
                      row_keys = ['rsmpF',  'cnnf', 'cgs','hyd_fine'],
                      rowLabels_d = None,
                      #pie_legend=True, 
                      box_fp=None, 
                      
                      log=None, out_dir=None, ofp=None,
 
                      fig_mat_kwargs=dict(figsize=None),
                      **kwargs):
        """matrix plot comparing methods for downscaling: rasters
 
        rows: simNames
        columns
            depth grid (w/ asset exposure)
            confusion grid (w/ inundation metrics)
            
            
        NOTE: uses the grid_keys from hp.rio_base.RioPlotr
            
        Pars
        --------
        metric_df: DataFrame
            inundation performance metrics
            rows: simulations/scenarios
            
        grids_ds: xr.DataSource
            grids w/ grid_key:
                'WSH' and 'CONFU'
 
            
 
            
        box_fp: str
            optional filepath to add a black focus box to the plot
        """
        #===========================================================================
        # setup
        #===========================================================================
        start = datetime.now() 
        #configure outputs
        if ofp is  None:
            if out_dir is None:
                out_dir = os.path.join(wrk_dir, 'outs', 'cnnf', 'post_plots')
         
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                
            ofp = os.path.join(out_dir, f'plot_inun_perf_{today_str}.{output_format}')
            
        if log is None: log  = get_log_stream('plot_inun_perf') #get the root logger
 
    
        #=======================================================================
        # defaults
        #=======================================================================
   
        log.info(f'on {str(metric_df.shape)}')
 
        font_size=matplotlib.rcParams['font.size']
        
        
        #get confusion style shortcuts
        cc_d = self.confusion_codes.copy()
        cval_d = self.confusion_val_d.copy()        
        confusion_color_d=self.confusion_color_d.copy()
            
        if rowLabels_d is None:
            rowLabels_d=dict()
        

        #spatial meta from dem for working with points
        #
         
        
        #bounding box
        if box_fp is None:
            focus_bbox=None
        else:
            
            focus_bbox, crs=get_bbox_and_crs(box_fp)
            log.info(f'using aoi from \'{os.path.basename(box_fp)}\'')
            #===================================================================
            # rmeta_d = get_meta(fp_df.iloc[0,0])
            # assert crs == rmeta_d['crs']
            #===================================================================
 
        #=======================================================================
        # setup figure
        #=======================================================================
        if row_keys is None:
            row_keys = grid_ds['tag'].values.tolist()            
        #true_tag= list(set(row_keys).difference(metric_df.index.values))[0]
            
 
        col_keys = ['WSH', 'CONFU']
            
  
        fig, ax_d = self.get_matrix_fig(row_keys, col_keys, logger=log, 
                                        set_ax_title=False, constrained_layout=True,
                                        **fig_mat_kwargs)
 
        #=======================================================================
        # plot grids------
        #=======================================================================
        focus_poly = None
        axImg_d = dict() #container for objects for colorbar
        #dep1_yet=False
        for rowk, d0 in ax_d.items():
            for colk, ax in d0.items():                
                gridk = colk.upper()
                if not rowk in axImg_d: axImg_d[rowk]=dict()
                #aname = nicknames_d2[rowk]
                aname=rowLabels_d[rowk]                
                log.debug(f'plot loop for {rowk}.{colk}.{gridk} ({aname})')
                
 
                #===============================================================
                # raster plot-----
                #===============================================================
                ar = grid_ds.loc[{'grid_key':gridk, 'tag':rowk}].values
                if np.any(np.isnan(ar)):
                    log.warning(f'got nulls on {rowk}.{colk}')
 
                
                log.info(f'plotting {rowk} x {colk} ({gridk})')
                
                axImg_d[rowk][colk] = self._ax_imshow(ax,ar, gridk=gridk)                    
 
 
                #===========================================================
                # focus box--------
                #===========================================================
                #if (colk=='c3') and (not focus_bbox is None) and (rowk==row_keys[0]): #first map only
                if colk==col_keys[-1] and rowk==row_keys[0] and not focus_bbox is None:
                    x, y = focus_bbox.exterior.coords.xy 
                    polygon_points = [[x[i], y[i]] for i in range(len(x))]                        
                                                        
                    focus_poly=ax.add_patch(
                        matplotlib.patches.Polygon(polygon_points, edgecolor='blue', facecolor='none', linewidth=1.0, linestyle='dashed')
                        )
                    
                    ax.legend([focus_poly], ['detail'], loc='upper right')
 
                #===========================================================
                # post format-------
                #===========================================================
                #hide labels
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                
        #=======================================================================
        # add metrics
        #=======================================================================
        for rowk, d0 in ax_d.items():
            for i, (colk, ax) in enumerate(d0.items()):  
                gridk=colk
                #add text
                if colk=='CONFU':
                    metric_df.columns
                    md = metric_df.loc[rowk, :].to_dict()
                    
                    md = {v:md[k] for k,v in self.metric_labels.items()}
 
 
 

                elif colk=='WSH':
                    self._add_scaleBar_northArrow(ax)
                    
                    da = grid_ds.loc[{'grid_key':gridk, 'tag':rowk}]
                    
                    md={'max':da.max().item(), 'mean':da.mean().item()}
                    
                else:
                    raise KeyError(gridk)
                
                
                ax.text(0.98, 0.05, get_dict_str(md, num_format = '{:.3f}'), transform=ax.transAxes, 
                            va='bottom', ha='right', fontsize=font_size, color='black',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.8),
                            )
 
 
        #=======================================================================
        # colorbar-------
        #=======================================================================
        for rowk, d0 in ax_d.items():
            for colk, ax in d0.items():
                #only last row 
                if not rowk==row_keys[-1]:
                    continue
                               
                gridk = colk
                            
                location, fmt, label, spacing = self._get_colorbar_pars_by_key(gridk)
                
                cbar = fig.colorbar(axImg_d[rowk][colk],
                                #cax=cax, 
 
                                orientation='horizontal',
                         ax=ax, location=location, # steal space from here
                         extend='both', #pointed ends
                         format=fmt, label=label,spacing=spacing,
                         shrink=0.8,
                         )
                
                #relabel
                if 'CONFU' == gridk: 
                    new_ticks = (cbar.get_ticks()[:-1] + cbar.get_ticks()[1:]) / 2
                    cbar.set_ticks(new_ticks,
                                    #[(101-1)/2+1, 101.5, (111-102)/2+102, 111.5], 
                                   labels = [{v:k for k,v in cc_d.items()}[k0] for k0 in cval_d.keys()] 
                                   )
                    
 
        #=======================================================================
        # post format-------
        #======================================================================= 
        for rowk, d0 in ax_d.items():
            for colk, ax in d0.items():
 
                
                #first col
                if colk==col_keys[0]:
                    if rowk in rowLabels_d:
                        rowlab = rowLabels_d[rowk]
                    else:
                        rowlab = rowk
                        
                    ax.set_ylabel(rowlab)
 
 
        #=======================================================================
        # wrap
        #=======================================================================
        
        
        fig.savefig(ofp, dpi = 300, format = output_format, transparent=True)
        
        log.info(f'finished and wrote figure to \n    {ofp}')
                    
        return ofp
    
    
    
 
   