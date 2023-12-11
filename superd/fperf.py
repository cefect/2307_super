'''
Created on Aug. 22, 2023

@author: cefect

skinny implementation of fperf
'''
import logging

import numpy as np
import pandas as pd
from hp.logr import get_log_stream

from parameters import confusion_codes


def get_confusion_cat(true_arB, pred_arB, 
                      confusion_codes={'TP':'TP', 'TN':'TN', 'FP':'FP', 'FN':'FN'},
                      ):
    """compute the confusion code for each element
    
    Parameters
    -----------
    confusion_codes: dict
        optional mapping for naming the 4 confusion categories
    """
    #start with dummy
    res_ar = np.full(true_arB.shape, np.nan)
    #true positives
    res_ar = np.where(
        np.logical_and(true_arB, pred_arB), 
        confusion_codes['TP'], res_ar)
    #true negatives
    res_ar = np.where(
        np.logical_and(np.invert(true_arB), np.invert(pred_arB)), 
        confusion_codes['TN'], res_ar)
    #false positives
    res_ar = np.where(
        np.logical_and(np.invert(true_arB), pred_arB), 
        confusion_codes['FP'], res_ar)
    #false negatives
    res_ar = np.where(
        np.logical_and(true_arB, np.invert(pred_arB)), 
        confusion_codes['FN'], res_ar)
    
    
    return res_ar 

class ValidateMask(object):
    """compute validation metrics for a inundation mask
    
    
    WARNING: use context management for multiple runs
    """ 
    
    confusion_ser=None
    
 
 
    
    def __init__(self,
                 confu_ar=None,
 
                 logger=None,
                  ):
        """
        
        Pars
        ------------
 
            
        true_inun_fp: str
            filepath to inundation (rlay or poly) to compare extents against
            
        pred_inun_fp: str
            filepath to predicted/modelled inundation (rlay) to evaluate
        """
        self.confu_ar=confu_ar
 
        #=======================================================================
        # pre init
        #=======================================================================
        if logger is None:
            logger = get_log_stream(level=logging.DEBUG)
            
        self.logger=logger
        
 
 
 
    #===========================================================================
    # grid inundation metrics------
    #===========================================================================
    def get_hitRate(self, **kwargs):
        """proportion of wet benchmark data that was replicated by the model"""
        # log, true_ar, pred_ar = self._func_setup_local('hitRate', **kwargs)
        
        cf_ser = self._confusion(**kwargs)
        
        return cf_ser['TP'] / (cf_ser['TP'] + cf_ser['FN'])
    
    def get_falseAlarms(self, **kwargs):
        # log, true_ar, pred_ar = self._func_setup_local('hitRate', **kwargs)
        
        cf_ser = self._confusion(**kwargs)
        
        return cf_ser['FP'] / (cf_ser['TP'] + cf_ser['FP'])
    
    def get_criticalSuccessIndex(self, **kwargs):
        """critical success index. accounts for both overprediction and underprediction"""
        
        cf_ser = self._confusion(**kwargs)
        
        return cf_ser['TP'] / (cf_ser['TP'] + cf_ser['FP'] + cf_ser['FN'])
    
    def get_errorBias(self, **kwargs):
        """indicates whether the model has a tendency toward overprediction or underprediction"""
        
        cf_ser = self._confusion(**kwargs)
        return cf_ser['FP'] / cf_ser['FN']
    
    def get_inundation_all(self, **kwargs):
        """convenience for getting all the inundation metrics
        NOT using notation from Wing (2017)
        """
        
        d = {
            'hitRate':self.get_hitRate(**kwargs),
            'falseAlarms':self.get_falseAlarms(**kwargs),
            'criticalSuccessIndex':self.get_criticalSuccessIndex(**kwargs),
            'errorBias':self.get_errorBias(**kwargs),
            }
        
        # add confusion codes
        d.update(self.confusion_ser.to_dict())
        
        #=======================================================================
        # #checks
        #=======================================================================
        assert set(d.keys()).symmetric_difference(
            ['hitRate', 'falseAlarms', 'criticalSuccessIndex', 'errorBias', 'TN', 'FP', 'FN', 'TP']
            ) == set()
            
        assert pd.Series(d).notna().all(), d
        
        self.logger.debug('computed all inundation metrics:\n    %s' % d)
        return d

    def get_confusion_grid(self,
                           confusion_codes=None, **kwargs):
        """generate confusion grid
        
        Parameters
        ----------
        confusion_codes: dict
            integer codes for confusion labels
        """
        #(true=wet=nonnull)
        log, true_arB, pred_arB = self._func_setup_local('confuGrid', **kwargs)
        
        if confusion_codes is None: confusion_codes = self.confusion_codes
 
        #build a confusion map (using the int codes)
        res_ar = get_confusion_cat(true_arB, pred_arB, confusion_codes=confusion_codes)
        
        #=======================================================================
        # check
        #=======================================================================
        if __debug__:
            """compare against our aspatial confusion generator"""
            cf_ser = self._confusion(true_mar=true_arB, pred_mar=pred_arB, **kwargs)
            
            # build a frame with the codes
            df1 = pd.Series(res_ar.ravel(), name='grid_counts').value_counts().to_frame().reset_index()            
            df1['index'] = df1['index'].astype(int) 
            df2 = df1.join(pd.Series({v:k for k, v in confusion_codes.items()}, name='codes'), on='index'
                           ).set_index('index')
                           
            # join the values from sklearn calcs
            df3 = df2.join(cf_ser.rename('sklearn_counts').reset_index().rename(columns={'index':'codes'}).set_index('codes'),
                     on='codes')
            
            #compare
            if not df3['grid_counts'].sum()==df3['sklearn_counts'].sum():
                raise AssertionError('mismatch in confusion counts')
            
            compare_bx = df3['grid_counts'] == df3['sklearn_counts']
            if not compare_bx.all():
                raise AssertionError('confusion count mismatch\n    %s' % compare_bx.to_dict())
            
        log.info('finished on %s' % str(res_ar.shape))
        
        return res_ar
 
    #===========================================================================
    # private helpers------
    #===========================================================================
    def _confusion(self, **kwargs):
        """retrieve or construct the wet/dry confusion series"""
        if self.confusion_ser is None:
 
            cser = pd.Series(self.confu_ar.ravel()).value_counts()
            
            #re-label index 
            cser.index = cser.index.map(lambda x: {v:k for k,v in confusion_codes.items()}[x])
            
            #add empties
            for k,v in confusion_codes.items():
                if not k in cser.index:
                    cser[k] = 0

            self.confusion_ser = cser.copy()
            
        return self.confusion_ser.copy()

 
        
        
    def __enter__(self):
        return self
    
    def __exit__(self,  *args,**kwargs):
        self.confusion_ser=None
        pass
 
 
        
   