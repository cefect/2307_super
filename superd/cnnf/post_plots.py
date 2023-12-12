"""main caller for CNNF post plots"""

 




#===============================================================================
# imports---------
#===============================================================================
import os
from datetime import datetime
import numpy as np

from hp.basic import get_dict_str, today_str
from hp.logr import get_log_stream
from hp.rio import RioPlotr
from definitions import wrk_dir
import matplotlib
import matplotlib.pyplot as plt





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
        #rmeta_d = get_meta(fp_df.iloc[0,0])
         
        
        #bounding box
        if box_fp is None:
            focus_bbox=None
        else:
            
            focus_bbox, crs=get_bbox_and_crs(box_fp)
            log.info(f'using aoi from \'{os.path.basename(box_fp)}\'')
            assert crs == rmeta_d['crs']
 
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
    
 
   